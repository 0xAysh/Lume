import { useState } from "react";
import { search, type SearchHit } from "./lib/commands";

// Walking-skeleton shell (BUILD.md M1): a single search bar over a bare grid.
// The adaptive cutoff, virtualized grid, filters, and previews are M4. This file
// deliberately holds no ranking/business logic — it renders what `search`
// returns (DESIGN §19).
export default function App() {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [status, setStatus] = useState<"idle" | "searching" | "error">("idle");

  async function runSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setStatus("searching");
    try {
      setHits(await search(query));
      setStatus("idle");
    } catch {
      setStatus("error");
    }
  }

  return (
    <main className="mx-auto flex h-full max-w-5xl flex-col gap-6 p-8 text-neutral-100">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Lume</h1>
        <p className="text-sm text-neutral-400">Describe a photo or video moment — it stays on your Mac.</p>
      </header>

      <form onSubmit={runSearch}>
        <input
          autoFocus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="a girl riding a bicycle"
          className="w-full rounded-lg border border-neutral-700 bg-neutral-900 px-4 py-3 text-base outline-none focus:border-neutral-500"
        />
      </form>

      <section className="flex-1 overflow-auto">
        {status === "error" && <p className="text-red-400">Search failed.</p>}
        {status !== "error" && hits.length === 0 && (
          <p className="text-neutral-500">
            {status === "searching" ? "Searching…" : "No results yet — indexing and search land in M1–M4."}
          </p>
        )}
        <div className="grid grid-cols-4 gap-2">
          {hits.map((hit) => (
            <img
              key={hit.fileId}
              src={hit.thumbUrl}
              alt=""
              className="aspect-square w-full rounded-md object-cover"
            />
          ))}
        </div>
      </section>
    </main>
  );
}
