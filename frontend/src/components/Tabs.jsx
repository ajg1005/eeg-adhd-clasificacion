export function Tabs({ activeTab, onTabChange, tabs }) {
  return (
    <nav className="tabs">
      {tabs.map((tab) => (
        <button
          className={activeTab === tab ? "tab-button active" : "tab-button"}
          key={tab}
          onClick={() => onTabChange(tab)}
          type="button"
        >
          {tab}
        </button>
      ))}
    </nav>
  );
}

