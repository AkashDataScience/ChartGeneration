const { useState, useEffect, useRef, useMemo } = React;

const NODE_WIDTH = 216;
const NODE_HEIGHT = 66;

function formatTokens(num) {
  const n = Number(num || 0);
  if (n >= 1000) return `${(n / 1000).toFixed(2)}k tks`;
  return `${n} tks`;
}

function getPortCoords(node, port) {
  const x = node.x;
  const y = node.y;
  switch (port) {
    case "top":
      return { x: x + NODE_WIDTH / 2, y };
    case "bottom":
      return { x: x + NODE_WIDTH / 2, y: y + NODE_HEIGHT };
    case "left":
      return { x, y: y + NODE_HEIGHT / 2 };
    case "right":
      return { x: x + NODE_WIDTH, y: y + NODE_HEIGHT / 2 };
    default:
      return { x: x + NODE_WIDTH / 2, y: y + NODE_HEIGHT };
  }
}

/**
 * Generates an orthogonal step path with clean rounded corners matching the screenshot's connectors.
 */
function buildOrthogonalPath(s, t, sourcePort, targetPort) {
  const dx = t.x - s.x;
  const dy = t.y - s.y;

  if (Math.abs(dx) < 2) {
    return `M ${s.x} ${s.y} L ${t.x} ${t.y}`;
  }

  if (sourcePort === "right" && targetPort === "left") {
    const midX = s.x + dx / 2;
    return `M ${s.x} ${s.y} L ${midX} ${s.y} L ${midX} ${t.y} L ${t.x} ${t.y}`;
  }

  if (sourcePort === "left" && targetPort === "top") {
    return `M ${s.x} ${s.y} L ${t.x} ${s.y} L ${t.x} ${t.y}`;
  }

  if (sourcePort === "right" && targetPort === "top") {
    const stepOut = s.x + 28;
    const midY = s.y + (t.y - s.y) * 0.45;
    if (t.x > s.x) {
      return `M ${s.x} ${s.y} L ${t.x} ${s.y} L ${t.x} ${t.y}`;
    }
    return `M ${s.x} ${s.y} L ${stepOut} ${s.y} L ${stepOut} ${midY} L ${t.x} ${midY} L ${t.x} ${t.y}`;
  }

  if (sourcePort === "bottom" && targetPort === "left") {
    return `M ${s.x} ${s.y} L ${s.x} ${t.y} L ${t.x} ${t.y}`;
  }

  if (sourcePort === "bottom" && targetPort === "right") {
    const outX = Math.max(s.x, t.x) + 46;
    const midY = s.y + 28;
    return `M ${s.x} ${s.y} L ${s.x} ${midY} L ${outX} ${midY} L ${outX} ${t.y} L ${t.x} ${t.y}`;
  }

  // Default top/bottom step path
  const midY = s.y + dy * 0.5;
  return `M ${s.x} ${s.y} L ${s.x} ${midY} L ${t.x} ${midY} L ${t.x} ${t.y}`;
}

/* ==================== SVG ICONS ==================== */
const Icons = {
  Runs: ({ active }) => (
    <svg className={`w-4 h-4 ${active ? "text-[#818cf8]" : "text-zinc-400"}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  ),
  Data: ({ active }) => (
    <svg className={`w-4 h-4 ${active ? "text-[#818cf8]" : "text-zinc-400"}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  ),
  Reports: ({ active }) => (
    <svg className={`w-4 h-4 ${active ? "text-[#818cf8]" : "text-zinc-400"}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  ),
  Bot: () => (
    <svg className="w-4 h-4 text-zinc-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <rect x="4" y="8" width="16" height="12" rx="3" />
      <path d="M12 4v4" />
      <circle cx="12" cy="3" r="1" />
      <circle cx="9" cy="14" r="1.2" fill="currentColor" />
      <circle cx="15" cy="14" r="1.2" fill="currentColor" />
    </svg>
  ),
  Search: () => (
    <svg className="w-3.5 h-3.5 text-zinc-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  ),
  Clock: () => (
    <svg className="w-3 h-3 text-zinc-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  ),
  Globe: () => (
    <svg className="w-3.5 h-3.5 text-zinc-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  ),
  SidebarToggle: () => (
    <svg className="w-4 h-4 text-zinc-500 hover:text-zinc-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <line x1="9" y1="3" x2="9" y2="21" />
    </svg>
  )
};

function rewriteReportImagePaths(mdText, runId) {
  if (!mdText) return "";
  return mdText.replace(/!\[([^\]]*)\]\((?!http|\/)(charts\/[^)]+)\)/g, `![$1](/reports/${runId}/$2)`);
}

/* ==================== MAIN APP COMPONENT ==================== */
function App() {
  const [activeNav, setActiveNav] = useState("runs"); // "runs" | "data" | "reports"
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Runs state
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [activeRunData, setActiveRunData] = useState(null);
  const [runFilter, setRunFilter] = useState("Manual"); // "All" | "Manual" | "Scheduled"
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [showNewRunModal, setShowNewRunModal] = useState(false);

  // New Run Form
  const [newRunDataset, setNewRunDataset] = useState("data/coffee_sales.csv");
  const [newRunGoal, setNewRunGoal] = useState("");
  const [newRunMaxCharts, setNewRunMaxCharts] = useState(3);
  const [creatingRun, setCreatingRun] = useState(false);

  // Data tab state
  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetName, setSelectedDatasetName] = useState("coffee_sales.csv");
  const [datasetProfile, setDatasetProfile] = useState(null);
  const [uploadingCsv, setUploadingCsv] = useState(false);

  // Reports tab state
  const [reportSubTab, setReportSubTab] = useState("markdown"); // "markdown" | "gallery" | "followup"
  const [followupInput, setFollowupInput] = useState("");
  const [followupHistory, setFollowupHistory] = useState([]);
  const [followupLoading, setFollowupLoading] = useState(false);

  // Canvas Pan & Zoom + Node Dragging
  const [pan, setPan] = useState({ x: 40, y: 20 });
  const [zoom, setZoom] = useState(0.95);
  const [locked, setLocked] = useState(false);
  const [customNodeCoords, setCustomNodeCoords] = useState({});
  const dragRef = useRef({ type: null, nodeId: null, startX: 0, startY: 0, origX: 0, origY: 0 });

  // Fetch initial runs and datasets
  const fetchRuns = async () => {
    try {
      const res = await fetch("/api/runs");
      const data = await res.json();
      const list = data.runs || [];
      setRuns(list);
      if (!selectedRunId && list.length > 0) {
        setSelectedRunId(list[0].run_id);
      }
    } catch (err) {
      console.error("Failed to fetch runs:", err);
    }
  };

  const fetchDatasets = async () => {
    try {
      const res = await fetch("/api/datasets");
      const data = await res.json();
      const list = data.datasets || [];
      setDatasets(list);
      if (list.length > 0 && !selectedDatasetName) {
        setSelectedDatasetName(list[0].name);
      }
    } catch (err) {
      console.error("Failed to fetch datasets:", err);
    }
  };

  const fetchRunDetails = async (runId) => {
    if (!runId) return;
    try {
      const res = await fetch(`/api/runs/${runId}`);
      if (res.ok) {
        const data = await res.json();
        setActiveRunData(data);
      }
    } catch (err) {
      console.error("Failed to fetch run details:", err);
    }
  };

  const fetchDatasetProfile = async (filename) => {
    if (!filename) return;
    try {
      const res = await fetch(`/api/datasets/${filename}/profile`);
      if (res.ok) {
        const data = await res.json();
        setDatasetProfile(data);
      }
    } catch (err) {
      console.error("Failed to fetch dataset profile:", err);
    }
  };

  useEffect(() => {
    fetchRuns();
    fetchDatasets();
  }, []);

  useEffect(() => {
    if (selectedRunId) {
      setCustomNodeCoords({});
      fetchRunDetails(selectedRunId);
    }
  }, [selectedRunId]);

  // Poll while any run is RUNNING
  useEffect(() => {
    const isRunning =
      activeRunData?.status === "RUNNING" || runs.some((r) => r.status === "RUNNING");
    if (!isRunning) return;
    const timer = setInterval(() => {
      fetchRuns();
      if (selectedRunId) fetchRunDetails(selectedRunId);
    }, 2000);
    return () => clearInterval(timer);
  }, [activeRunData?.status, runs, selectedRunId]);

  useEffect(() => {
    if (selectedDatasetName) {
      fetchDatasetProfile(selectedDatasetName);
    }
  }, [selectedDatasetName]);

  // Compute DAG nodes with user drag overrides
  const dagNodes = useMemo(() => {
    const rawNodes = activeRunData?.dag?.nodes || [];
    return rawNodes.map((n) => {
      const override = customNodeCoords[n.id];
      return override ? { ...n, x: override.x, y: override.y } : n;
    });
  }, [activeRunData, customNodeCoords]);

  const dagEdges = activeRunData?.dag?.edges || [];

  const selectedNode = useMemo(
    () => dagNodes.find((n) => n.id === selectedNodeId) || null,
    [dagNodes, selectedNodeId]
  );

  // Canvas mouse handlers
  const handleCanvasMouseDown = (e) => {
    if (e.target.closest(".dag-node-card") || e.target.closest(".dag-controls")) return;
    dragRef.current = {
      type: "canvas",
      startX: e.clientX,
      startY: e.clientY,
      origX: pan.x,
      origY: pan.y
    };
  };

  const handleNodeMouseDown = (e, node) => {
    e.stopPropagation();
    setSelectedNodeId(node.id);
    if (locked) return;
    dragRef.current = {
      type: "node",
      nodeId: node.id,
      startX: e.clientX,
      startY: e.clientY,
      origX: node.x,
      origY: node.y
    };
  };

  const handleMouseMove = (e) => {
    const d = dragRef.current;
    if (!d.type) return;
    if (d.type === "canvas") {
      setPan({
        x: d.origX + (e.clientX - d.startX),
        y: d.origY + (e.clientY - d.startY)
      });
    } else if (d.type === "node") {
      const dx = (e.clientX - d.startX) / zoom;
      const dy = (e.clientY - d.startY) / zoom;
      setCustomNodeCoords((prev) => ({
        ...prev,
        [d.nodeId]: { x: Math.round(d.origX + dx), y: Math.round(d.origY + dy) }
      }));
    }
  };

  const handleMouseUp = () => {
    dragRef.current = { type: null };
  };

  const handleLaunchRun = async (e) => {
    e.preventDefault();
    setCreatingRun(true);
    try {
      const res = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_path: newRunDataset,
          user_goal: newRunGoal,
          max_charts: Number(newRunMaxCharts) || 3
        })
      });
      const data = await res.json();
      if (data.run_id) {
        setShowNewRunModal(false);
        setNewRunGoal("");
        await fetchRuns();
        setSelectedRunId(data.run_id);
        setActiveNav("runs");
      }
    } catch (err) {
      console.error("Error creating run:", err);
    } finally {
      setCreatingRun(false);
    }
  };

  const handleUploadCsv = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadingCsv(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/datasets/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.name) {
        await fetchDatasets();
        setSelectedDatasetName(data.name);
      }
    } catch (err) {
      console.error("Upload failed:", err);
    } finally {
      setUploadingCsv(false);
    }
  };

  const handleSendFollowup = async (e) => {
    e.preventDefault();
    if (!followupInput.trim() || !selectedRunId) return;
    const q = followupInput.trim();
    setFollowupInput("");
    setFollowupHistory((prev) => [...prev, { role: "user", text: q }]);
    setFollowupLoading(true);
    try {
      const res = await fetch(`/api/runs/${selectedRunId}/followup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q })
      });
      const data = await res.json();
      setFollowupHistory((prev) => [
        ...prev,
        {
          role: "agent",
          text: data.answer || "Completed request.",
          chart: data.chart || null
        }
      ]);
      await fetchRunDetails(selectedRunId);
    } catch (err) {
      setFollowupHistory((prev) => [
        ...prev,
        { role: "agent", text: `Error: ${err.message}` }
      ]);
    } finally {
      setFollowupLoading(false);
    }
  };

  const filteredRuns = useMemo(() => {
    return runs.filter((r) => {
      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase();
      return (
        (r.user_goal || "").toLowerCase().includes(q) ||
        (r.dataset_path || "").toLowerCase().includes(q) ||
        (r.run_id || "").toLowerCase().includes(q)
      );
    });
  }, [runs, searchQuery]);

  return (
    <div
      className="flex h-screen w-screen bg-[#070709] text-zinc-200 overflow-hidden"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {/* ==================== 1. LEFT CORE NAVIGATION SIDEBAR ==================== */}
      <aside
        className={`${
          sidebarCollapsed ? "w-14" : "w-44"
        } flex flex-col justify-between bg-[#090a0d] border-r border-[#181920] transition-all duration-150 select-none z-20`}
      >
        <div>
          {/* Top Brand + Collapse Icon */}
          <div className="h-12 px-3.5 flex items-center justify-between border-b border-[#15161c]">
            {!sidebarCollapsed && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-[#6366f1] shadow-[0_0_8px_#6366f1]" />
                <span className="text-xs font-semibold tracking-wider text-zinc-200 uppercase">
                  ChartAgent
                </span>
              </div>
            )}
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="p-1 rounded hover:bg-zinc-800/60 transition"
              title="Toggle Sidebar"
            >
              <Icons.SidebarToggle />
            </button>
          </div>

          {/* CORE Section (ONLY Runs, Data, Reports as requested) */}
          <div className="px-2.5 py-4">
            {!sidebarCollapsed && (
              <div className="px-2 mb-2 text-[10px] font-semibold tracking-widest text-zinc-500 uppercase">
                CORE
              </div>
            )}
            <nav className="space-y-1">
              {[
                { id: "runs", label: "Runs", Icon: Icons.Runs },
                { id: "data", label: "Data", Icon: Icons.Data },
                { id: "reports", label: "Reports", Icon: Icons.Reports }
              ].map(({ id, label, Icon }) => {
                const active = activeNav === id;
                return (
                  <button
                    key={id}
                    onClick={() => setActiveNav(id)}
                    className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-xs font-medium transition relative ${
                      active
                        ? "bg-[#19192b] text-white border border-[#2d2e4a]"
                        : "text-zinc-400 hover:text-zinc-200 hover:bg-[#111218]"
                    }`}
                  >
                    {active && (
                      <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 bg-[#818cf8] rounded-r" />
                    )}
                    <Icon active={active} />
                    {!sidebarCollapsed && <span>{label}</span>}
                  </button>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Bottom Status Footer */}
        {!sidebarCollapsed && (
          <div className="p-3 border-t border-[#15161c] text-[11px] text-zinc-500 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
              MCP v2 Ready
            </span>
            <span className="font-mono text-[10px] text-zinc-600">v2.4</span>
          </div>
        )}
      </aside>

      {/* ==================== 2. SECONDARY CONTEXTUAL LIST PANEL ==================== */}
      <section className="w-[315px] flex flex-col bg-[#0b0c10] border-r border-[#181920] z-10">
        {activeNav === "runs" && (
          <>
            {/* Search runs + '+' button */}
            <div className="p-3 flex items-center gap-2 border-b border-[#16171f]">
              <div className="flex-1 flex items-center gap-2 bg-[#12131a] border border-[#222430] rounded-lg px-2.5 py-1.5">
                <Icons.Search />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search runs..."
                  className="bg-transparent text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none w-full"
                />
              </div>
              <button
                onClick={() => setShowNewRunModal(true)}
                className="w-7 h-7 flex items-center justify-center rounded-lg bg-[#151722] hover:bg-[#232638] border border-[#282b3e] text-zinc-300 hover:text-white text-sm font-medium transition"
                title="Start New Analysis Run"
              >
                +
              </button>
            </div>

            {/* Space: Global + Filter Tabs (All | Manual | Scheduled) */}
            <div className="px-3.5 py-2 flex items-center justify-between border-b border-[#16171f] text-[11px]">
              <div className="flex items-center gap-1.5 text-zinc-400">
                <Icons.Globe />
                <span>Space: Global</span>
              </div>
              <div className="flex items-center gap-3">
                {["All", "Manual", "Scheduled"].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setRunFilter(tab)}
                    className={`pb-0.5 transition ${
                      runFilter === tab
                        ? "text-white font-semibold border-b border-white"
                        : "text-zinc-500 hover:text-zinc-300"
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>
            </div>

            {/* Run Cards List */}
            <div className="flex-1 overflow-y-auto p-2.5 space-y-2">
              {filteredRuns.map((run) => {
                const isSelected = run.run_id === selectedRunId;
                const isRunning = run.status === "RUNNING";
                const isFailed = run.status === "FAILED";
                return (
                  <div
                    key={run.run_id}
                    onClick={() => setSelectedRunId(run.run_id)}
                    className={`p-3.5 rounded-xl cursor-pointer transition border ${
                      isSelected
                        ? "bg-[#0e111d] border-[#2e3b6e] shadow-[0_0_20px_rgba(59,76,145,0.18)]"
                        : "bg-[#0d0e13] border-[#181a24] hover:border-[#262938]"
                    }`}
                  >
                    <p
                      className={`text-xs leading-relaxed mb-3 ${
                        isSelected ? "text-[#8da2fb] font-medium" : "text-zinc-300"
                      }`}
                    >
                      {run.user_goal}
                    </p>
                    <div className="flex items-center justify-between text-[11px] text-zinc-500">
                      <span className="flex items-center gap-1">
                        <Icons.Clock />
                        {run.created_at}
                      </span>
                      <span className="font-mono text-[10.5px] text-zinc-400">
                        {formatTokens(run.total_tokens)}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider uppercase flex items-center gap-1 border ${
                          isRunning
                            ? "bg-amber-950/50 text-amber-400 border-amber-700/50 animate-pulse"
                            : isFailed
                            ? "bg-red-950/50 text-red-400 border-red-800/50"
                            : "bg-[#0c2617] text-[#22c55e] border-[#164e2b]"
                        }`}
                      >
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            isRunning
                              ? "bg-amber-400"
                              : isFailed
                              ? "bg-red-400"
                              : "bg-[#22c55e]"
                          }`}
                        />
                        {run.status}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}

        {activeNav === "data" && (
          <>
            <div className="p-3 flex items-center justify-between border-b border-[#16171f]">
              <span className="text-xs font-semibold text-zinc-300 tracking-wide">
                DATASETS ({datasets.length})
              </span>
              <label className="px-2.5 py-1 rounded-lg bg-[#1d2238] hover:bg-[#272e4b] border border-[#343e67] text-[#93a8ff] text-xs font-medium cursor-pointer transition">
                {uploadingCsv ? "Uploading..." : "+ Upload CSV"}
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleUploadCsv}
                  className="hidden"
                />
              </label>
            </div>
            <div className="flex-1 overflow-y-auto p-2.5 space-y-2">
              {datasets.map((ds) => {
                const active = ds.name === selectedDatasetName;
                return (
                  <div
                    key={ds.name}
                    onClick={() => setSelectedDatasetName(ds.name)}
                    className={`p-3.5 rounded-xl cursor-pointer transition border ${
                      active
                        ? "bg-[#0e111d] border-[#2e3b6e]"
                        : "bg-[#0d0e13] border-[#181a24] hover:border-[#262938]"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span
                        className={`text-xs font-semibold ${
                          active ? "text-[#8da2fb]" : "text-zinc-200"
                        }`}
                      >
                        {ds.name}
                      </span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800/80 text-zinc-400">
                        {ds.size_kb} KB
                      </span>
                    </div>
                    <div className="text-[11px] text-zinc-500 flex items-center justify-between">
                      <span>{ds.rows.toLocaleString()} rows</span>
                      <span>{ds.cols} columns</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}

        {activeNav === "reports" && (
          <>
            <div className="p-3 flex items-center justify-between border-b border-[#16171f]">
              <span className="text-xs font-semibold text-zinc-300 tracking-wide">
                GENERATED REPORTS ({runs.length})
              </span>
            </div>
            <div className="flex-1 overflow-y-auto p-2.5 space-y-2">
              {runs.map((run) => {
                const isSelected = run.run_id === selectedRunId;
                return (
                  <div
                    key={run.run_id}
                    onClick={() => setSelectedRunId(run.run_id)}
                    className={`p-3.5 rounded-xl cursor-pointer transition border ${
                      isSelected
                        ? "bg-[#0e111d] border-[#2e3b6e]"
                        : "bg-[#0d0e13] border-[#181a24] hover:border-[#262938]"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[11px] font-mono text-emerald-400">
                        {run.dataset_path}
                      </span>
                      <span className="text-[10px] text-zinc-500">{run.created_at}</span>
                    </div>
                    <p
                      className={`text-xs line-clamp-2 mb-2.5 ${
                        isSelected ? "text-[#8da2fb] font-medium" : "text-zinc-300"
                      }`}
                    >
                      {run.user_goal}
                    </p>
                    <div className="flex items-center justify-between text-[10.5px] text-zinc-500">
                      <span>{run.charts_count} Reflected Charts</span>
                      <span className="font-mono">{formatTokens(run.total_tokens)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </section>

      {/* ==================== 3. MAIN WORKSPACE ==================== */}
      <main className="flex-1 relative flex flex-col bg-[#070709] overflow-hidden">
        {/* ---------- VIEW A: RUNS (INTERACTIVE DAG FLOW CANVAS) ---------- */}
        {activeNav === "runs" && (
          <div
            className="flex-1 relative dag-canvas-bg overflow-hidden cursor-grab active:cursor-grabbing"
            onMouseDown={handleCanvasMouseDown}
          >
            {/* Top-Left Canvas Controls Pill (+, -, Fit, Lock) matching screenshot */}
            <div className="dag-controls absolute top-4 left-4 z-20 flex flex-col bg-[#111218] border border-[#222430] rounded-lg overflow-hidden shadow-lg">
              <button
                onClick={() => setZoom((z) => Math.min(z + 0.12, 1.8))}
                className="w-8 h-8 flex items-center justify-center text-zinc-400 hover:text-white hover:bg-[#1c1e29] border-b border-[#20222e] text-sm"
                title="Zoom In"
              >
                +
              </button>
              <button
                onClick={() => setZoom((z) => Math.max(z - 0.12, 0.45))}
                className="w-8 h-8 flex items-center justify-center text-zinc-400 hover:text-white hover:bg-[#1c1e29] border-b border-[#20222e] text-sm"
                title="Zoom Out"
              >
                −
              </button>
              <button
                onClick={() => {
                  setPan({ x: 40, y: 20 });
                  setZoom(0.95);
                  setCustomNodeCoords({});
                }}
                className="w-8 h-8 flex items-center justify-center text-zinc-400 hover:text-white hover:bg-[#1c1e29] border-b border-[#20222e] text-xs"
                title="Fit & Reset View"
              >
                ⛶
              </button>
              <button
                onClick={() => setLocked((l) => !l)}
                className={`w-8 h-8 flex items-center justify-center text-xs ${
                  locked
                    ? "text-amber-400 bg-amber-950/30"
                    : "text-zinc-400 hover:text-white hover:bg-[#1c1e29]"
                }`}
                title={locked ? "Unlock Node Dragging" : "Lock Node Positions"}
              >
                {locked ? "🔒" : "🔓"}
              </button>
            </div>

            {/* Top-Right Floating Quick Actions */}
            <div className="dag-controls absolute top-4 right-4 z-20 flex items-center gap-2">
              {activeRunData && (
                <button
                  onClick={() => setActiveNav("reports")}
                  className="px-3 py-1.5 rounded-lg bg-[#141622] hover:bg-[#1e2235] border border-[#2c324c] text-xs font-medium text-[#93a8ff] flex items-center gap-1.5 shadow-lg transition"
                >
                  <span>Open Full Report</span>
                  <span>→</span>
                </button>
              )}
            </div>

            {/* Pan & Zoom Transformed Container */}
            <div
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: "0 0",
                width: "1400px",
                height: "950px"
              }}
              className="relative"
            >
              {/* SVG Connectors Layer */}
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{ overflow: "visible" }}
              >
                {dagEdges.map((edge) => {
                  const sNode = dagNodes.find((n) => n.id === edge.source);
                  const tNode = dagNodes.find((n) => n.id === edge.target);
                  if (!sNode || !tNode) return null;
                  const s = getPortCoords(sNode, edge.sourcePort);
                  const t = getPortCoords(tNode, edge.targetPort);
                  const d = buildOrthogonalPath(s, t, edge.sourcePort, edge.targetPort);
                  const isRunning =
                    sNode.status === "RUNNING" || tNode.status === "RUNNING";
                  return (
                    <g key={edge.id}>
                      <path
                        d={d}
                        fill="none"
                        stroke={isRunning ? "#fbbf24" : "#6c6f7e"}
                        strokeWidth={isRunning ? "1.8" : "1.35"}
                        strokeLinejoin="round"
                        className={isRunning ? "edge-running" : ""}
                      />
                    </g>
                  );
                })}
              </svg>

              {/* Agent Nodes Layer */}
              {dagNodes.map((node) => {
                const isSelected = selectedNodeId === node.id;
                const isRunning = node.status === "RUNNING";
                const isFailed = node.status === "FAILED";
                const isPending = node.status === "PENDING";

                return (
                  <div
                    key={node.id}
                    onMouseDown={(e) => handleNodeMouseDown(e, node)}
                    style={{
                      left: `${node.x}px`,
                      top: `${node.y}px`,
                      width: `${NODE_WIDTH}px`,
                      height: `${NODE_HEIGHT}px`
                    }}
                    className={`dag-node-card absolute rounded-2xl bg-[#111216] px-3.5 py-2.5 flex items-center gap-3 cursor-pointer transition-shadow select-none border ${
                      isSelected
                        ? "border-[#818cf8] shadow-[0_0_24px_rgba(129,140,248,0.28)]"
                        : isRunning
                        ? "border-amber-400/80 shadow-[0_0_22px_rgba(251,191,36,0.25)]"
                        : "border-[#252730] hover:border-[#3b3f4e] shadow-[0_8px_24px_rgba(0,0,0,0.65)]"
                    }`}
                  >
                    {/* Connection Ports (Top, Bottom, Left, Right) matching screenshot */}
                    <span className="w-2.5 h-2.5 rounded-full bg-[#7c7f8e] border-2 border-[#070709] absolute -top-1.5 left-1/2 -translate-x-1/2" />
                    <span className="w-2.5 h-2.5 rounded-full bg-[#7c5dfa] border-2 border-[#070709] absolute -bottom-1.5 left-1/2 -translate-x-1/2" />
                    <span className="w-2.5 h-2.5 rounded-full bg-[#9ca3af] border-2 border-[#070709] absolute top-1/2 -left-1.5 -translate-y-1/2" />
                    <span className="w-2.5 h-2.5 rounded-full bg-[#9ca3af] border-2 border-[#070709] absolute top-1/2 -right-1.5 -translate-y-1/2" />

                    {/* Status Dot + Robot Icon */}
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          isRunning
                            ? "bg-amber-400 animate-ping"
                            : isFailed
                            ? "bg-red-500"
                            : isPending
                            ? "bg-zinc-600"
                            : "bg-[#22c55e]"
                        }`}
                      />
                      <div className="w-7 h-7 rounded-lg bg-[#181920] border border-[#262936] flex items-center justify-center">
                        <Icons.Bot />
                      </div>
                    </div>

                    {/* Role & Agent Label */}
                    <div className="flex flex-col min-w-0">
                      <span className="text-[9.5px] font-semibold tracking-widest text-zinc-400 uppercase truncate">
                        {node.role}
                      </span>
                      <span
                        className={`text-xs font-semibold truncate ${
                          isRunning
                            ? "text-amber-400"
                            : isFailed
                            ? "text-red-400"
                            : isPending
                            ? "text-zinc-500"
                            : "text-[#22c55e]"
                        }`}
                      >
                        {node.label}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Bottom-Left Hint */}
            <div className="absolute bottom-4 left-4 text-[11px] text-zinc-500 bg-[#0d0e13]/90 border border-[#1d1f2b] px-3 py-1.5 rounded-lg pointer-events-none">
              Click any agent node to inspect Tree-of-Thoughts, Pandas verification, & V1/V2 charts
            </div>

            {/* ---------- SLIDE-OVER NODE INSPECTOR DRAWER ---------- */}
            {selectedNode && (
              <div className="dag-controls absolute top-0 right-0 bottom-0 w-[460px] bg-[#0c0d12] border-l border-[#1f222e] shadow-2xl z-30 flex flex-col">
                <div className="p-4 border-b border-[#1b1d28] flex items-center justify-between">
                  <div>
                    <div className="text-[10px] font-mono uppercase tracking-widest text-[#818cf8]">
                      {selectedNode.role} • {selectedNode.status}
                    </div>
                    <h3 className="text-sm font-bold text-white mt-0.5">
                      {selectedNode.label} — {selectedNode.subtitle}
                    </h3>
                  </div>
                  <button
                    onClick={() => setSelectedNodeId(null)}
                    className="w-7 h-7 rounded-lg bg-[#161822] hover:bg-[#222534] text-zinc-400 hover:text-white flex items-center justify-center text-sm"
                  >
                    ✕
                  </button>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs">
                  <div className="p-3 rounded-lg bg-[#12131c] border border-[#202332] text-zinc-300 font-medium">
                    {selectedNode.details?.stage}
                  </div>

                  {/* PlannerAgent Details */}
                  {selectedNode.id === "planner" && (
                    <>
                      <div>
                        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                          Dataset Summary
                        </h4>
                        <p className="p-3 rounded-lg bg-[#101118] border border-[#1d202d] text-zinc-300 leading-relaxed">
                          {selectedNode.details.dataset_summary || "Profiling dataset..."}
                        </p>
                      </div>
                      <div>
                        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                          Mandatory Cleaning Rules
                        </h4>
                        <p className="p-3 rounded-lg bg-[#101118] border border-[#1d202d] text-emerald-300 font-mono text-[11px] leading-relaxed">
                          {selectedNode.details.cleaning_rules || "Auditing columns..."}
                        </p>
                      </div>
                      {selectedNode.details.analytical_questions?.length > 0 && (
                        <div>
                          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                            Formulated Hypotheses
                          </h4>
                          <ul className="space-y-1.5">
                            {selectedNode.details.analytical_questions.map((q, i) => (
                              <li
                                key={i}
                                className="p-2.5 rounded-lg bg-[#101118] border border-[#1d202d] text-zinc-300"
                              >
                                <span className="text-[#818cf8] font-bold mr-1.5">H{i + 1}:</span>
                                {q}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  )}

                  {/* StatisticalAgent Details */}
                  {selectedNode.id === "statistical" && (
                    <>
                      {(selectedNode.details.findings || []).map((f, i) => (
                        <div
                          key={i}
                          className="p-3 rounded-xl bg-[#101118] border border-[#1d202d] space-y-2"
                        >
                          <div className="font-semibold text-[#8da2fb]">
                            Q{i + 1}: {f.question}
                          </div>
                          <div className="text-zinc-400 text-[11px]">{f.Quantitative_evidence || f.quantitative_evidence}</div>
                          {f.markdown_table && (
                            <div
                              className="markdown-body"
                              dangerouslySetInnerHTML={{
                                __html: marked.parse(f.markdown_table)
                              }}
                            />
                          )}
                          <div className="text-emerald-400 text-[11px] font-medium">
                            Conclusion: {f.conclusion}
                          </div>
                        </div>
                      ))}
                    </>
                  )}

                  {/* ChartAgent V1 Details */}
                  {selectedNode.id.startsWith("chart_gen_") && (
                    <>
                      <div>
                        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1">
                          Tree-of-Thoughts & Candidate Plan Selection
                        </h4>
                        <pre className="p-3 rounded-lg bg-[#101118] border border-[#1d202d] text-zinc-300 font-mono text-[11px] whitespace-pre-wrap leading-relaxed">
                          {selectedNode.details.thought_v1 || "Generating candidate plans..."}
                        </pre>
                      </div>
                      {selectedNode.details.chart_v1_url && (
                        <div>
                          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                            Generated Draft Chart (V1)
                          </h4>
                          <img
                            src={selectedNode.details.chart_v1_url}
                            alt="Chart V1"
                            className="w-full rounded-lg border border-[#262936] bg-white p-1"
                          />
                        </div>
                      )}
                      {selectedNode.details.code_v1 && (
                        <div>
                          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1">
                            Decomposed Python Code (V1)
                          </h4>
                          <pre className="p-3 rounded-lg bg-[#090a0f] border border-[#1d202d] text-blue-300 font-mono text-[10.5px] overflow-x-auto">
                            {selectedNode.details.code_v1}
                          </pre>
                        </div>
                      )}
                    </>
                  )}

                  {/* ReflectorAgent V2 Details */}
                  {selectedNode.id.startsWith("chart_ref_") && (
                    <>
                      <div>
                        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1">
                          Visual Critique on V1
                        </h4>
                        <p className="p-3 rounded-lg bg-[#101118] border border-[#1d202d] text-amber-300 leading-relaxed whitespace-pre-wrap">
                          {selectedNode.details.critique || "Evaluating V1 visual aesthetics..."}
                        </p>
                      </div>
                      <div>
                        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1">
                          Refinement Applied (V2)
                        </h4>
                        <p className="p-3 rounded-lg bg-[#101118] border border-[#1d202d] text-emerald-300 leading-relaxed">
                          {selectedNode.details.feedback}
                        </p>
                      </div>
                      {selectedNode.details.chart_v2_url && (
                        <div>
                          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                            Reflected Final Chart (V2)
                          </h4>
                          <img
                            src={selectedNode.details.chart_v2_url}
                            alt="Chart V2"
                            className="w-full rounded-lg border border-[#262936] bg-white p-1"
                          />
                        </div>
                      )}
                    </>
                  )}

                  {/* Distiller / Formatter Details */}
                  {(selectedNode.id === "distiller" || selectedNode.id === "formatter") && (
                    <>
                      <div>
                        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                          Executive Summary
                        </h4>
                        <p className="p-3 rounded-lg bg-[#101118] border border-[#1d202d] text-zinc-200 leading-relaxed">
                          {selectedNode.details.executive_summary || "Synthesizing findings..."}
                        </p>
                      </div>
                      <button
                        onClick={() => setActiveNav("reports")}
                        className="w-full py-2.5 rounded-lg bg-[#4f46e5] hover:bg-[#4338ca] text-white font-semibold text-xs transition"
                      >
                        View Full Synthesized Report in Reports Tab →
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ---------- VIEW B: DATA (DATASET PROFILER & EXPLORER) ---------- */}
        {activeNav === "data" && (
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {datasetProfile ? (
              <>
                <div className="flex items-center justify-between border-b border-[#191b24] pb-4">
                  <div>
                    <span className="text-[11px] font-mono text-[#818cf8] uppercase tracking-wider">
                      AUTOMATED DATA PROFILER
                    </span>
                    <h1 className="text-lg font-bold text-white mt-0.5">
                      {datasetProfile.path}
                    </h1>
                  </div>
                  <button
                    onClick={() => {
                      setNewRunDataset(datasetProfile.path);
                      setShowNewRunModal(true);
                    }}
                    className="px-4 py-2 rounded-lg bg-[#4f46e5] hover:bg-[#4338ca] text-white text-xs font-semibold transition shadow-lg"
                  >
                    + Run End-to-End Analysis on {datasetProfile.name}
                  </button>
                </div>

                {/* 4 KPI Cards */}
                <div className="grid grid-cols-4 gap-4">
                  {[
                    { label: "Total Rows", value: datasetProfile.rows.toLocaleString() },
                    { label: "Columns", value: datasetProfile.cols },
                    { label: "Memory Usage", value: `${datasetProfile.memory_kb} KB` },
                    { label: "Duplicate Rows", value: datasetProfile.duplicates }
                  ].map((kpi, i) => (
                    <div
                      key={i}
                      className="p-4 rounded-xl bg-[#0e1017] border border-[#1d202d]"
                    >
                      <div className="text-[11px] text-zinc-500 uppercase tracking-wider">
                        {kpi.label}
                      </div>
                      <div className="text-xl font-bold text-white mt-1 font-mono">
                        {kpi.value}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Column Schema & Quality Table */}
                <div className="rounded-xl bg-[#0e1017] border border-[#1d202d] overflow-hidden">
                  <div className="px-4 py-3 border-b border-[#1d202d] text-xs font-semibold text-zinc-300">
                    Column Schema, Missing Values & Outlier Audit
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-[#131520] text-zinc-400 border-b border-[#1d202d]">
                        <tr>
                          <th className="py-2.5 px-4">Column</th>
                          <th className="py-2.5 px-4">Data Type</th>
                          <th className="py-2.5 px-4">Missing</th>
                          <th className="py-2.5 px-4">Unique</th>
                          <th className="py-2.5 px-4">IQR Outliers</th>
                          <th className="py-2.5 px-4">Distribution / Top Values</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#171923]">
                        {datasetProfile.columns.map((c) => (
                          <tr key={c.column} className="hover:bg-[#12141e]">
                            <td className="py-2.5 px-4 font-mono font-semibold text-[#8da2fb]">
                              {c.column}
                            </td>
                            <td className="py-2.5 px-4 font-mono text-zinc-400">
                              {c.dtype}
                            </td>
                            <td className="py-2.5 px-4">
                              {c.missing > 0 ? (
                                <span className="text-amber-400 font-mono">
                                  {c.missing} ({c.missing_pct}%)
                                </span>
                              ) : (
                                <span className="text-emerald-400 font-mono">0 (0%)</span>
                              )}
                            </td>
                            <td className="py-2.5 px-4 font-mono text-zinc-300">
                              {c.unique.toLocaleString()}
                            </td>
                            <td className="py-2.5 px-4 font-mono text-zinc-300">
                              {c.outliers}
                            </td>
                            <td className="py-2.5 px-4 text-zinc-400 font-mono text-[11px]">
                              {c.summary}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Sample Rows Table */}
                <div className="rounded-xl bg-[#0e1017] border border-[#1d202d] overflow-hidden">
                  <div className="px-4 py-3 border-b border-[#1d202d] text-xs font-semibold text-zinc-300">
                    Sample Records (First 25 Rows)
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs font-mono">
                      <thead className="bg-[#131520] text-zinc-400 border-b border-[#1d202d]">
                        <tr>
                          {datasetProfile.columns.map((c) => (
                            <th key={c.column} className="py-2 px-3">
                              {c.column}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#171923]">
                        {datasetProfile.sample_rows.map((row, rIdx) => (
                          <tr key={rIdx} className="hover:bg-[#12141e]">
                            {datasetProfile.columns.map((c) => (
                              <td key={c.column} className="py-2 px-3 text-zinc-300">
                                {String(row[c.column])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-zinc-500 text-sm">Loading dataset profile...</div>
            )}
          </div>
        )}

        {/* ---------- VIEW C: REPORTS (MARKDOWN REPORT + V1/V2 GALLERY + FOLLOWUP) ---------- */}
        {activeNav === "reports" && (
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Report Header & Sub-Tabs */}
            <div className="px-6 py-3.5 bg-[#0b0c10] border-b border-[#181920] flex items-center justify-between">
              <div>
                <div className="text-[10.5px] font-mono text-emerald-400">
                  {activeRunData?.dataset_path} • {activeRunData?.created_at}
                </div>
                <h2 className="text-sm font-bold text-white truncate max-w-xl">
                  {activeRunData?.user_goal}
                </h2>
              </div>
              <div className="flex items-center gap-2 bg-[#12131a] p-1 rounded-lg border border-[#222430]">
                {[
                  { id: "markdown", label: "Executive Report (report.md)" },
                  { id: "gallery", label: "V1 vs V2 Reflection Gallery" },
                  { id: "followup", label: "Interactive Follow-Up Q&A" }
                ].map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setReportSubTab(t.id)}
                    className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
                      reportSubTab === t.id
                        ? "bg-[#232741] text-white"
                        : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Report Content Area */}
            <div className="flex-1 overflow-y-auto p-6">
              {reportSubTab === "markdown" && activeRunData && (
                <div className="max-w-4xl mx-auto rounded-2xl bg-[#0c0d13] border border-[#1d202d] p-8 shadow-xl">
                  <div
                    className="markdown-body select-text"
                    dangerouslySetInnerHTML={{
                      __html: marked.parse(
                        rewriteReportImagePaths(
                          activeRunData.full_markdown_report ||
                            activeRunData.executive_summary ||
                            "Report is currently being synthesized...",
                          activeRunData.run_id
                        )
                      )
                    }}
                  />
                </div>
              )}

              {reportSubTab === "gallery" && activeRunData && (
                <div className="max-w-6xl mx-auto space-y-6">
                  {(activeRunData.chart_results || []).map((c, idx) => (
                    <div
                      key={c.chart_id || idx}
                      className="rounded-2xl bg-[#0c0d13] border border-[#1d202d] p-6 space-y-4"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="text-[10px] font-mono uppercase tracking-widest text-[#818cf8]">
                            CHART #{idx + 1} • {c.chart_id}
                          </span>
                          <h3 className="text-base font-bold text-white">{c.title}</h3>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-6">
                        <div className="space-y-2">
                          <div className="text-xs font-semibold text-zinc-400">
                            Draft V1 (Initial Generation)
                          </div>
                          <img
                            src={`/${String(c.chart_v1).replace(/^\/+/, "")}`}
                            alt="V1"
                            className="w-full rounded-xl border border-[#252836] bg-white p-1.5"
                          />
                        </div>
                        <div className="space-y-2">
                          <div className="text-xs font-semibold text-emerald-400">
                            Reflected V2 (Post-Visual Critique)
                          </div>
                          <img
                            src={`/${String(c.chart_v2).replace(/^\/+/, "")}`}
                            alt="V2"
                            className="w-full rounded-xl border border-emerald-800/50 bg-white p-1.5"
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 pt-2 text-xs">
                        <div className="p-3.5 rounded-xl bg-[#11131c] border border-[#202334]">
                          <div className="text-[10px] font-semibold uppercase text-amber-400 mb-1">
                            ReflectorAgent Visual Critique
                          </div>
                          <p className="text-zinc-300 leading-relaxed whitespace-pre-wrap">
                            {c.critique}
                          </p>
                        </div>
                        <div className="p-3.5 rounded-xl bg-[#11131c] border border-[#202334]">
                          <div className="text-[10px] font-semibold uppercase text-emerald-400 mb-1">
                            Refinements Applied in V2
                          </div>
                          <p className="text-zinc-300 leading-relaxed">{c.feedback}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {reportSubTab === "followup" && activeRunData && (
                <div className="max-w-3xl mx-auto flex flex-col h-full">
                  <div className="flex-1 space-y-4 mb-4">
                    <div className="p-4 rounded-xl bg-[#10121b] border border-[#202436] text-xs text-zinc-300">
                      Ask any quantitative follow-up question about{" "}
                      <code className="text-[#8da2fb]">{activeRunData.dataset_path}</code> or
                      request a new/modified chart (e.g.,{" "}
                      <em>
                        "What is the exact total revenue for Cortado?" or "Plot a pie chart of
                        cash vs card transactions"
                      </em>
                      ).
                    </div>
                    {followupHistory.map((item, idx) => (
                      <div
                        key={idx}
                        className={`p-4 rounded-xl text-xs leading-relaxed ${
                          item.role === "user"
                            ? "bg-[#191d30] border border-[#2e365a] text-[#9cb0ff] ml-12"
                            : "bg-[#0e1017] border border-[#1e2130] text-zinc-200 mr-12"
                        }`}
                      >
                        <div className="text-[10px] font-mono uppercase text-zinc-500 mb-1">
                          {item.role === "user" ? "You" : "FollowUpAgent"}
                        </div>
                        <div className="whitespace-pre-wrap">{item.text}</div>
                        {item.chart && (
                          <img
                            src={`/${String(item.chart.chart_v2).replace(/^\/+/, "")}`}
                            alt="Follow-up chart"
                            className="mt-3 rounded-lg border border-[#262936] bg-white p-1 max-h-80"
                          />
                        )}
                      </div>
                    ))}
                  </div>
                  <form onSubmit={handleSendFollowup} className="flex gap-2">
                    <input
                      type="text"
                      value={followupInput}
                      onChange={(e) => setFollowupInput(e.target.value)}
                      placeholder="Ask a follow-up question or request a chart modification..."
                      className="flex-1 bg-[#11131c] border border-[#25293a] rounded-xl px-4 py-3 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-[#6366f1]"
                    />
                    <button
                      type="submit"
                      disabled={followupLoading}
                      className="px-5 py-3 rounded-xl bg-[#4f46e5] hover:bg-[#4338ca] disabled:opacity-50 text-white text-xs font-semibold transition"
                    >
                      {followupLoading ? "Running..." : "Send"}
                    </button>
                  </form>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* ==================== NEW RUN MODAL ==================== */}
      {showNewRunModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-2xl bg-[#0e1017] border border-[#252838] shadow-2xl overflow-hidden">
            <div className="px-5 py-4 border-b border-[#1c1e2b] flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">
                Start New Autonomous Analysis Run
              </h3>
              <button
                onClick={() => setShowNewRunModal(false)}
                className="text-zinc-400 hover:text-white text-sm"
              >
                ✕
              </button>
            </div>
            <form onSubmit={handleLaunchRun} className="p-5 space-y-4 text-xs">
              <div>
                <label className="block text-zinc-400 mb-1.5 font-medium">
                  Select Dataset
                </label>
                <select
                  value={newRunDataset}
                  onChange={(e) => setNewRunDataset(e.target.value)}
                  className="w-full bg-[#141621] border border-[#262a3d] rounded-lg px-3 py-2 text-zinc-200 focus:outline-none focus:border-[#6366f1]"
                >
                  {datasets.map((d) => (
                    <option key={d.path} value={d.path}>
                      {d.path} ({d.rows.toLocaleString()} rows)
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-zinc-400 mb-1.5 font-medium">
                  Analysis Goal / Prompt
                </label>
                <textarea
                  rows="3"
                  value={newRunGoal}
                  onChange={(e) => setNewRunGoal(e.target.value)}
                  placeholder="e.g., Calculate exact revenue for Cortado, compare cash vs card trends, and identify peak hours..."
                  className="w-full bg-[#141621] border border-[#262a3d] rounded-lg p-3 text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-[#6366f1]"
                />
              </div>
              <div>
                <label className="block text-zinc-400 mb-1.5 font-medium">
                  Number of Visualizations (V1 → V2 Reflection)
                </label>
                <select
                  value={newRunMaxCharts}
                  onChange={(e) => setNewRunMaxCharts(Number(e.target.value))}
                  className="w-full bg-[#141621] border border-[#262a3d] rounded-lg px-3 py-2 text-zinc-200 focus:outline-none"
                >
                  <option value={1}>1 Chart (Fastest)</option>
                  <option value={2}>2 Charts</option>
                  <option value={3}>3 Charts (Comprehensive)</option>
                </select>
              </div>
              <div className="pt-2 flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowNewRunModal(false)}
                  className="px-4 py-2 rounded-lg bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={creatingRun}
                  className="px-4 py-2 rounded-lg bg-[#4f46e5] hover:bg-[#4338ca] text-white font-semibold"
                >
                  {creatingRun ? "Launching..." : "Launch Multi-Agent Run"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
