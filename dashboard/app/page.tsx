export default function Home() {
  return (
    <main className="p-8 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold">Phoenix Multi‑Agent SOC</h1>
      <p className="mt-2">Paste triage JSON below to visualize.</p>
      <textarea id="input" className="w-full h-64 mt-4 border p-2" placeholder="{...triage json...}" />
      <button className="mt-4 border px-4 py-2" onClick={() => {
        const raw = (document.getElementById('input') as HTMLTextAreaElement).value;
        try {
          const j = JSON.parse(raw);
          alert(JSON.stringify({ severity: j?.output?.severity, summary: j?.output?.summary }, null, 2));
        } catch (e) { alert('Invalid JSON'); }
      }}>Visualize</button>
    </main>
  );
}
