#!/usr/bin/env node

import { copyFileSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const APPIUM_PORT = Number(process.env.LUME_APPIUM_PORT ?? 4723);
const APPIUM_URL = `http://127.0.0.1:${APPIUM_PORT}`;
const BUNDLE_ID = process.env.LUME_BUNDLE_ID ?? "app.lume.desktop";
const APP_BUNDLE = resolve(
  process.env.LUME_APP_BUNDLE ?? join(ROOT, "src-tauri/target/debug/bundle/macos/Lume.app"),
);

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});

async function main() {
  if (process.platform !== "darwin") {
    console.error("macOS Appium smoke requires macOS. Lume v1 app-window automation runs only on macOS runners.");
    if (process.env.LUME_MACOS_APP_REQUIRED) process.exitCode = 78;
    return;
  }

  assertCommand("appium", "Run `npm install` and `npm run setup:macos-appium` first.");

  if (!existsSync(APP_BUNDLE) || !process.env.LUME_SKIP_TAURI_BUILD) {
    run("npx", ["tauri", "build", "--debug", "--bundles", "app", "--no-sign"], {
      cwd: ROOT,
      env: process.env,
    });
  }

  if (!existsSync(APP_BUNDLE)) {
    throw new Error(`Tauri app bundle not found at ${APP_BUNDLE}`);
  }

  const fixture = createFixture();
  let appium;
  let sessionId;
  try {
    appium = spawn("appium", ["--port", String(APPIUM_PORT), "--log-level", process.env.LUME_APPIUM_LOG_LEVEL ?? "warn"], {
      cwd: ROOT,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    appium.stdout.on("data", (chunk) => process.stdout.write(`[appium] ${chunk}`));
    appium.stderr.on("data", (chunk) => process.stderr.write(`[appium] ${chunk}`));

    await waitForServer(appium, 30_000);
    sessionId = await createSession(fixture);

    const indexButton = await findOne(sessionId, [
      ["accessibility id", "Index watched folder"],
      ["predicate string", 'label == "Index watched folder" OR title == "Index watched folder" OR value == "Index watched folder"'],
      ["xpath", '//*[@label="Index watched folder" or @title="Index watched folder" or @value="Index watched folder"]'],
    ]);
    await click(sessionId, indexButton);

    await waitUntil(
      async () => (await source(sessionId)).includes("Indexed"),
      60_000,
      "indexing did not finish or did not become visible",
    );

    const searchField = await findOne(sessionId, [
      ["accessibility id", "Search query"],
      [
        "predicate string",
        'label == "Search query" OR placeholderValue == "Search query" OR placeholderValue == "a girl riding a bicycle"',
      ],
      ["class name", "XCUIElementTypeTextField"],
      ["xpath", '//*[@label="Search query" or @placeholderValue="Search query" or @placeholderValue="a girl riding a bicycle"]'],
    ]);
    await click(sessionId, searchField);
    await type(sessionId, searchField, "anything\n");

    const resultCount = await waitUntil(
      async () => {
        const results = await findMany(sessionId, [
          ["predicate string", 'label BEGINSWITH "Search result"'],
          ["xpath", '//*[starts-with(@label, "Search result")]'],
        ]);
        return results.length > 0 ? results.length : false;
      },
      30_000,
      "search did not render any accessible result images",
    );

    console.log(`macOS Appium smoke passed with ${resultCount} rendered result(s).`);
  } finally {
    if (sessionId) {
      await request("DELETE", `/session/${sessionId}`).catch(() => {});
    }
    if (appium && !appium.killed) {
      appium.kill("SIGTERM");
    }
    rmSync(fixture.root, { recursive: true, force: true });
  }
}

function createFixture() {
  const root = mkdtempSync(join(tmpdir(), "lume-macos-appium-"));
  const home = join(root, "home");
  const watch = join(root, "watch");
  mkdirSync(home, { recursive: true });
  mkdirSync(watch, { recursive: true });

  const icon = join(ROOT, "src-tauri/icons/icon.png");
  copyFileSync(icon, join(watch, "sunlit-kitchen.png"));
  copyFileSync(icon, join(watch, "night-window.jpg"));
  writeFileSync(join(watch, "broken.jpg"), "not an image; the fake embedder owns this smoke");

  return { root, home, watch };
}

async function createSession(fixture) {
  const response = await request("POST", "/session", {
    capabilities: {
      alwaysMatch: {
        platformName: "Mac",
        "appium:automationName": "Mac2",
        "appium:bundleId": BUNDLE_ID,
        "appium:appPath": APP_BUNDLE,
        "appium:noReset": false,
        "appium:skipAppKill": false,
        "appium:serverStartupTimeout": 180_000,
        "appium:environment": {
          HOME: fixture.home,
          LUME_WATCH_FOLDER: fixture.watch,
          LUME_SIDECAR_FAKE_EMBEDDER: "1",
        },
      },
    },
  });
  return response.value?.sessionId ?? response.sessionId;
}

async function findOne(sessionId, locators) {
  for (const [using, value] of locators) {
    try {
      const response = await request("POST", `/session/${sessionId}/element`, { using, value });
      return unwrapElement(response.value);
    } catch {
      // Try the next locator. WebKit accessibility trees vary across macOS versions.
    }
  }
  throw new Error(`element not found; tried ${locators.map(([, value]) => value).join(" | ")}`);
}

async function findMany(sessionId, locators) {
  for (const [using, value] of locators) {
    try {
      const response = await request("POST", `/session/${sessionId}/elements`, { using, value });
      const elements = response.value.map(unwrapElement);
      if (elements.length > 0) return elements;
    } catch {
      // Try the next locator.
    }
  }
  return [];
}

async function click(sessionId, elementId) {
  await request("POST", `/session/${sessionId}/element/${elementId}/click`, {});
}

async function type(sessionId, elementId, value) {
  await request("POST", `/session/${sessionId}/element/${elementId}/value`, {
    text: value,
    value: [...value],
  });
}

async function source(sessionId) {
  const response = await request("GET", `/session/${sessionId}/source`);
  return response.value;
}

async function waitForServer(child, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`Appium exited early with status ${child.exitCode}`);
    }
    try {
      await request("GET", "/status");
      return;
    } catch {
      await sleep(250);
    }
  }
  throw new Error("Appium did not become ready");
}

async function request(method, path, body) {
  const response = await fetch(`${APPIUM_URL}${path}`, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const textBody = await response.text();
  const json = textBody ? JSON.parse(textBody) : {};
  if (!response.ok) {
    throw new Error(`${method} ${path} failed: ${textBody}`);
  }
  return json;
}

async function waitUntil(fn, timeoutMs, message) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await fn();
    if (value) return value;
    await sleep(500);
  }
  throw new Error(message);
}

function unwrapElement(value) {
  return value["element-6066-11e4-a52e-4f735466cecf"] ?? value.ELEMENT;
}

function assertCommand(command, message) {
  const check = spawnSync(command, ["--version"], { stdio: "ignore" });
  if (check.error?.code === "ENOENT") {
    throw new Error(`${command} is not installed. ${message}`);
  }
}

function run(command, args, options) {
  const result = spawnSync(command, args, { ...options, stdio: "inherit" });
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed with status ${result.status}`);
  }
}

function sleep(ms) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}
