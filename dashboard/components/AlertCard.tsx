export function AlertCard({ severity, summary }: { severity: string; summary: string; }) {
  return (
    <div className="border p-4 rounded-lg mb-2">
      <div><strong>Severity:</strong> {severity}</div>
      <div className="mt-1">{summary}</div>
    </div>
  );
}
