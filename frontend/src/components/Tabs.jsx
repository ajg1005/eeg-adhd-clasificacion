export function Tabs({ activeTab, disabled = false, onTabChange, tabs }) {
  return (
    <nav className="tabs">
      {tabs.map((tab) => (
        <button
          className={activeTab === tab ? "tab-button active" : "tab-button"}
          disabled={disabled && activeTab !== tab}
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
