// Example: Using the Natural Language Query API in a React Component
// This can be integrated into executive-dashboard.jsx

import { useState } from 'react';

interface QueryResult {
  summary: string;
  data: Array<{
    id: string;
    assetId: string;
    score: number;
    severity: string;
    region: string;
    lastUpdated: string;
  }>;
  recommendations?: string[];
  trends?: Array<{
    date: string;
    avgScore: number;
    count: number;
  }>;
}

export function useNaturalLanguageQuery() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const query = async (
    question: string,
    context?: {
      region?: string;
      days?: number;
      userRole?: string;
    }
  ): Promise<QueryResult | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          context: context || {},
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Query failed');
      }

      return data.result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      console.error('Query error:', err);
      return null;
    } finally {
      setLoading(false);
    }
  };

  return { query, loading, error };
}

// Example usage in a component:
export function QueryExample() {
  const { query, loading, error } = useNaturalLanguageQuery();
  const [result, setResult] = useState<QueryResult | null>(null);

  const handleQuery = async () => {
    const response = await query('What are the top 3 risks in Asia-Pacific?', {
      region: 'Asia-Pacific',
      days: 7,
    });

    if (response) {
      setResult(response);
    }
  };

  return (
    <div>
      <button onClick={handleQuery} disabled={loading}>
        {loading ? 'Querying...' : 'Get Top Risks'}
      </button>

      {error && <div className="error">{error}</div>}

      {result && (
        <div>
          <h3>Summary</h3>
          <p>{result.summary}</p>

          <h3>Risks ({result.data.length})</h3>
          {result.data.map((risk) => (
            <div key={risk.id} className="risk-card">
              <div className="risk-header">
                <strong>{risk.assetId}</strong>
                <span className={`severity-${risk.severity}`}>
                  {risk.severity}
                </span>
              </div>
              <div>Score: {risk.score}/100</div>
              <div>Region: {risk.region}</div>
            </div>
          ))}

          {result.recommendations && result.recommendations.length > 0 && (
            <>
              <h3>Recommendations</h3>
              <ul>
                {result.recommendations.map((rec, i) => (
                  <li key={i}>{rec}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// Example integration with existing executive-dashboard.jsx:
// 
// 1. Import the hook:
//    import { useNaturalLanguageQuery } from './useNaturalLanguageQuery';
//
// 2. Use in your component:
//    const { query, loading } = useNaturalLanguageQuery();
//
// 3. Replace mock responses with real queries:
//    const handleSubmit = async (e) => {
//      e.preventDefault();
//      if (!query.trim()) return;
//
//      setMessages(prev => [...prev, { type: 'user', content: query }]);
//
//      const result = await query(query, { region: selectedRegion });
//      
//      if (result) {
//        setMessages(prev => [...prev, {
//          type: 'assistant',
//          content: result.summary,
//          data: result.data,
//          recommendations: result.recommendations
//        }]);
//      }
//
//      setQuery('');
//    };
