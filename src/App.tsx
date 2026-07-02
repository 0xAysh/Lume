import { useEffect, useRef, useState } from "react";
import { indexStatus, search, startIndex, type IndexStatus, type SearchHit } from "./lib/commands";

// Walking-skeleton shell (BUILD.md M1): a single search bar over a bare grid,
// plus a way to trigger indexing and see its status. The adaptive cutoff,
// virtualized grid, filters, and previews are M4. This file deliberately
// holds no ranking/business logic — it renders what `search` returns
// (DESIGN §19).
export default function App() {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [status, setStatus] = useState<"idle" | "searching" | "error">("idle");
  const [index, setIndex] = useState<IndexStatus>({ phase: "idle", done: 0, total: 0 });
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function runSearch(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const formQuery = new FormData(e.currentTarget).get("query");
    const nextQuery = typeof formQuery === "string" ? formQuery : query;
    if (!nextQuery.trim()) return;
    setQuery(nextQuery);
    setStatus("searching");
    try {
      setHits(await search(nextQuery));
      setStatus("idle");
    } catch {
      setStatus("error");
    }
  }

  async function onStartIndex() {
    try {
      await startIndex();
    } catch {
      // Most likely "already in progress" — the poll below still reflects
      // the true backend state either way.
    }
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      const next = await indexStatus();
      setIndex(next);
      if (next.phase === "idle" || next.phase === "error") {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    }, 1000);
  }

  return (
    <main className="mx-auto flex h-full max-w-5xl flex-col gap-6 p-8 text-neutral-100">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Lume</h1>
          <p className="text-sm text-neutral-400">Describe a photo or video moment — it stays on your Mac.</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <button
            type="button"
            onClick={onStartIndex}
            className="rounded-lg border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm hover:border-neutral-500"
          >
            Index watched folder
          </button>
          <p className="text-xs text-neutral-500">
            {index.phase === "idle" && index.total === 0 && "Not indexed yet"}
            {index.phase === "scanning" && "Scanning…"}
            {index.phase === "indexing" && `Indexing ${index.done}/${index.total}`}
            {index.phase === "idle" && index.total > 0 && `Indexed ${index.done}/${index.total}`}
            {index.phase === "error" && "Indexing error"}
          </p>
        </div>
      </header>

      <form onSubmit={runSearch} className="flex gap-2">
        <input
          autoFocus
          name="query"
          aria-label="Search query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="a girl riding a bicycle"
          className="w-full rounded-lg border border-neutral-700 bg-neutral-900 px-4 py-3 text-base outline-none focus:border-neutral-500"
        />
        <button
          type="submit"
          aria-label="Search"
          className="rounded-lg border border-neutral-700 bg-neutral-900 px-4 py-3 text-sm hover:border-neutral-500"
        >
          Search
        </button>
      </form>

      <section className="flex-1 overflow-auto">
        {status === "error" && <p className="text-red-400">Search failed.</p>}
        {status !== "error" && hits.length === 0 && (
          <p className="text-neutral-500">
            {status === "searching" ? "Searching…" : "No results yet — index a folder, then search."}
          </p>
        )}
        <div className="grid grid-cols-4 gap-2">
          {hits.map((hit) => (
            <img
              key={hit.fileId}
              src={hit.thumbUrl}
              alt={`Search result ${hit.fileId}`}
              className="aspect-square w-full rounded-md object-cover"
            />
          ))}
        </div>
      </section>
    </main>
  );
}
