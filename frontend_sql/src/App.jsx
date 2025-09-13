import React, { useState } from 'react';
import './App.css';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';

function App() {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingLoading, setTrainingLoading] = useState(false);

  const API_BASE_URL = 'http://localhost:8000';

  const handleAskQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim() }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResponse(data);
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'Failed to get response from server');
    } finally {
      setLoading(false);
    }
  };

  const handleTrainModel = async () => {
    setTrainingLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/train_forecast`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setTrainingStatus(data);
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'Failed to train model');
    } finally {
      setTrainingLoading(false);
    }
  };

  const renderIntentBadge = (intent) => {
    let className = 'badge';
    let icon = '';

    switch (intent) {
      case 'Historical':
        className += ' badge-primary';
        icon = '';
        break;
      case 'Forecasting':
        className += ' badge-secondary';
        icon = '';
        break;
      case 'Unknown':
        className += ' badge-error';
        icon = '';
        break;
      default:
        className += ' badge-default';
    }

    return <span className={className}>{icon} {intent || 'Unknown'}</span>;
  };

  const renderTableData = (columns, rows) => {
    if (!rows || rows.length === 0) {
      return <div className="no-data">No data available</div>;
    }

    return (
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col, index) => (
                <th key={index}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {columns.map((col, colIndex) => (
                  <td key={colIndex}>
                    {row[col] !== null && row[col] !== undefined
                      ? String(row[col])
                      : <span className="null-value">null</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // 🔥 Chart rendering (line, bar, pie, forecasting)
  const renderChart = (response) => {
    if (!response || !response.chart) return null;

    // Historical Queries
    if (response.intent === 'Historical' && response.result?.rows) {
      const data = response.result.rows;
      const [xKey, yKey] = response.result.columns;

      if (response.chart === 'line') {
        return (
          <LineChart width={600} height={300} data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={xKey} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey={yKey} stroke="#8884d8" />
          </LineChart>
        );
      }

      if (response.chart === 'bar') {
        return (
          <BarChart width={600} height={300} data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={xKey} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey={yKey} fill="#82ca9d" />
          </BarChart>
        );
      }

      if (response.chart === 'pie') {
        const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A28CFE', '#FF6666'];
        return (
          <PieChart width={600} height={400}>
            <Tooltip />
            <Legend />
            <Pie
              data={data}
              dataKey={yKey}
              nameKey={xKey}
              cx="50%"
              cy="50%"
              outerRadius={150}
              label
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        );
      }
    }

    // Forecasting Queries
    if (response.intent === 'Forecasting' && Array.isArray(response.result)) {
      return (
        <LineChart width={600} height={300} data={response.result}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="ds" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="yhat" stroke="#8884d8" />
          <Line type="monotone" dataKey="yhat_lower" stroke="#82ca9d" />
          <Line type="monotone" dataKey="yhat_upper" stroke="#ff7300" />
        </LineChart>
      );
    }

    return null;
  };

  return (
    <div className="app">
      <div className="container">
        <div className="header">
          <div className="header-content">
            <h1> Data Query</h1>
            <p>Ask questions about your Northwind database data or request sales forecasts</p>
          </div>
        </div>

        <div className="main-content">
          {/* Question Input */}
          <div className="query-section">
            <div className="input-group">
              <input
                type="text"
                placeholder="e.g., Show me total sales by month, or Predict sales for next quarter"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion()}
                disabled={loading}
                className="question-input"
              />
              <button
                onClick={handleAskQuestion}
                disabled={loading || !question.trim()}
                className="ask-button"
              >
                {loading ? ' Asking...' : ' Ask Question'}
              </button>
            </div>

            {/* Train Model */}
            <div className="training-section">
              {/* <button
                onClick={handleTrainModel}
                disabled={trainingLoading}
                className="train-button"
              >
                {trainingLoading ? '⏳ Training...' : '🤖 Train Forecast Model'}
              </button> */}

              {trainingStatus && (
                <div className="success-message">
                  Model trained successfully with {trainingStatus.rows} rows of data.
                  Date range: {trainingStatus.date_range?.start} to {trainingStatus.date_range?.end}
                </div>
              )}
            </div>
          </div>

          {error && <div className="error-message">Error: {error}</div>}

          {/* Response Section */}
          {response && (
            <div className="response-section">
              <h2>Query Results</h2>

              <div className="response-info">
                <p><strong>Question:</strong> {response.question}</p>
                <p><strong>Intent:</strong> {renderIntentBadge(response.intent)}</p>
              </div>

              {/* {response.message && (
                <div className="message-info">💡 {response.message}</div>
              )} */}

              {response.sql && response.intent !== 'Forecasting' && (
                <div className="sql-section">
                  <details>
                    <summary>View Generated SQL</summary>
                    <pre className="sql-code">{response.sql}</pre>
                  </details>
                </div>
              )}

              {/* Forecast Summary */}
              {response.intent === 'Forecasting' && response.forecast_summary && (
                <div className="forecast-summary">
                  <h3> Forecast Summary</h3>
                  <div className="summary-content">
                    {response.forecast_summary.split('\n').map((line, index) => (
                      <p key={index}>{line}</p>
                    ))}
                  </div>
                </div>
              )}

              {/*  Render Charts */}
              {renderChart(response)}

              {/* Historical Table Fallback */}
              {response.intent === 'Historical' && response.result?.columns && (
                renderTableData(response.result.columns, response.result.rows)
              )}
            </div>
          )}

          {/* Welcome Examples */}
          {!response && !loading && (
            <div className="welcome-section">
              <p>Ask a question to see results here. Examples:</p>
              <div className="example-chips">
                <span className="example-chip" onClick={() => setQuestion("Show total sales by month")}>
                  Show total sales by month
                </span>
                <span className="example-chip" onClick={() => setQuestion("List top 5 products by revenue")}>
                  List top 5 products by revenue
                </span>
                <span className="example-chip" onClick={() => setQuestion("Predict sales for next quarter")}>
                  Predict sales for next quarter
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
