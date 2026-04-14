const state = {
  allStocks: [],
  stocks: [],
  sectors: [],
  selectedTicker: null,
};

const elements = {
  searchInput: document.getElementById("searchInput"),
  sectorSelect: document.getElementById("sectorSelect"),
  stockList: document.getElementById("stockList"),
  visibleCount: document.getElementById("visibleCount"),
  refreshStamp: document.getElementById("refreshStamp"),
  dataSource: document.getElementById("dataSource"),
  refreshSchedule: document.getElementById("refreshSchedule"),
  topIdeas: document.getElementById("topIdeas"),
  detailTicker: document.getElementById("detailTicker"),
  detailCompany: document.getElementById("detailCompany"),
  detailMeta: document.getElementById("detailMeta"),
  interestingBadges: document.getElementById("interestingBadges"),
  detailComposite: document.getElementById("detailComposite"),
  detailChange: document.getElementById("detailChange"),
  detailPrice: document.getElementById("detailPrice"),
  detailVolume: document.getElementById("detailVolume"),
  detailMarketCap: document.getElementById("detailMarketCap"),
  detailSetupLabel: document.getElementById("detailSetupLabel"),
  detailWhyNow: document.getElementById("detailWhyNow"),
  signalBars: document.getElementById("signalBars"),
  detailThesis: document.getElementById("detailThesis"),
};

function formatMoney(value) {
  if (value == null) return "--";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: value >= 100 ? 0 : 2,
  }).format(value);
}

function formatNumber(value) {
  if (value == null) return "--";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
}

function formatPercent(value) {
  if (value == null) return "--";
  const formatted = `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
  return formatted;
}

function scoreToPct(score) {
  return `${Math.round((score || 0) * 100)}%`;
}

async function loadStocks() {
  const search = elements.searchInput.value.trim().toLowerCase();
  const sector = elements.sectorSelect.value.trim().toLowerCase();
  const filtered = state.allStocks.filter((stock) => {
    const matchesSearch = !search
      || stock.ticker.toLowerCase().includes(search)
      || stock.company.toLowerCase().includes(search)
      || stock.industry.toLowerCase().includes(search);
    const matchesSector = !sector || stock.sector.toLowerCase() === sector;
    return matchesSearch && matchesSector;
  });

  state.stocks = filtered;
  elements.visibleCount.textContent = filtered.length;
  renderSectorOptions();
  renderStockList();

  const nextTicker = state.selectedTicker && filtered.some((stock) => stock.ticker === state.selectedTicker)
    ? state.selectedTicker
    : filtered[0]?.ticker;

  if (nextTicker) {
    renderDetail(state.allStocks.find((stock) => stock.ticker === nextTicker));
    state.selectedTicker = nextTicker;
    renderStockList();
  } else {
    clearDetail();
  }
}

function renderSectorOptions() {
  const current = elements.sectorSelect.value;
  const options = ['<option value="">All sectors</option>']
    .concat(state.sectors.map((sector) => `<option value="${sector}">${sector}</option>`));
  elements.sectorSelect.innerHTML = options.join("");
  elements.sectorSelect.value = current;
}

function renderTopIdeas() {
  const top = state.stocks.slice(0, 5);
  elements.topIdeas.innerHTML = top.map((stock) => `
    <button class="top-idea" data-top-ticker="${stock.ticker}">
      <strong>${stock.ticker} • ${scoreToPct(stock.signals.composite_score)}</strong>
      <span>${stock.setup_label}</span>
      <span class="${(stock.change_pct || 0) >= 0 ? "positive" : "negative"}">${formatPercent(stock.change_pct || 0)}</span>
    </button>
  `).join("");

  Array.from(elements.topIdeas.querySelectorAll("[data-top-ticker]")).forEach((node) => {
    node.addEventListener("click", () => selectStock(node.dataset.topTicker));
  });
}

function renderStockList() {
  if (!state.stocks.length) {
    elements.stockList.innerHTML = `<div class="stock-item"><div class="company">No stocks matched the current search.</div></div>`;
    return;
  }

  elements.stockList.innerHTML = state.stocks.map((stock) => {
    const isActive = stock.ticker === state.selectedTicker ? "active" : "";
    const changeClass = (stock.change_pct || 0) >= 0 ? "positive" : "negative";
    return `
      <button class="stock-item ${isActive}" data-ticker="${stock.ticker}">
        <div class="stock-item-top">
          <span class="ticker">${stock.ticker}</span>
          <span class="stock-chip">${scoreToPct(stock.signals.composite_score)}</span>
        </div>
        <div class="company">${stock.company}</div>
        <div class="stock-item-bottom">
          <span class="stock-chip">${stock.sector}</span>
          <span class="${changeClass}">${formatPercent(stock.change_pct || 0)}</span>
        </div>
      </button>
    `;
  }).join("");

  Array.from(elements.stockList.querySelectorAll("[data-ticker]")).forEach((node) => {
    node.addEventListener("click", () => selectStock(node.dataset.ticker));
  });
  renderTopIdeas();
}

async function selectStock(ticker) {
  const stock = state.allStocks.find((item) => item.ticker === ticker);
  if (!stock) return;
  state.selectedTicker = stock.ticker;
  renderStockList();
  renderDetail(stock);
}

function renderDetail(stock) {
  elements.detailTicker.textContent = stock.ticker;
  elements.detailCompany.textContent = stock.company;
  elements.detailMeta.textContent = `${stock.sector} • ${stock.industry} • ${stock.country}`;
  elements.interestingBadges.innerHTML = (stock.interesting_badges || [])
    .map((badge) => `<span class="stock-chip">${badge}</span>`)
    .join("");
  elements.detailComposite.textContent = scoreToPct(stock.signals.composite_score);
  elements.detailChange.textContent = formatPercent(stock.change_pct);
  elements.detailChange.className = (stock.change_pct || 0) >= 0 ? "positive" : "negative";
  elements.detailPrice.textContent = formatMoney(stock.price);
  elements.detailVolume.textContent = `${formatNumber(stock.volume)} shares`;
  elements.detailMarketCap.textContent = `${formatNumber(stock.market_cap_b)}B`;
  elements.detailSetupLabel.textContent = stock.setup_label;
  elements.detailWhyNow.textContent = stock.why_now || "This stock stands out for a mix of liquidity, momentum, and valuation context.";

  const labels = [
    ["Momentum", stock.signals.momentum_score],
    ["Volume", stock.signals.volume_score],
    ["Liquidity", stock.signals.liquidity_score],
    ["Valuation", stock.signals.valuation_score],
    ["Size fit", stock.signals.size_score],
    ["Technical blend", stock.signals.composite_score],
  ];

  elements.signalBars.innerHTML = labels.map(([label, score]) => `
    <div class="signal-row">
      <div class="signal-name">${label}</div>
      <div class="signal-track"><div class="signal-fill" style="width:${scoreToPct(score)}"></div></div>
      <div class="signal-value">${scoreToPct(score)}</div>
    </div>
  `).join("");

  elements.detailThesis.innerHTML = (stock.thesis || [])
    .map((line) => `<li>${line}</li>`)
    .join("");
}

function clearDetail() {
  elements.detailTicker.textContent = "No match";
  elements.detailCompany.textContent = "No stock selected";
  elements.detailMeta.textContent = "Try broadening the search.";
  elements.interestingBadges.innerHTML = "";
  elements.detailComposite.textContent = "--";
  elements.detailChange.textContent = "--";
  elements.detailPrice.textContent = "--";
  elements.detailVolume.textContent = "--";
  elements.detailMarketCap.textContent = "--";
  elements.detailSetupLabel.textContent = "Monitor";
  elements.detailWhyNow.textContent = "Pick a stock from the list to see its current angle.";
  elements.signalBars.innerHTML = "";
  elements.detailThesis.innerHTML = "";
}

elements.searchInput.addEventListener("input", () => loadStocks());
elements.sectorSelect.addEventListener("change", () => loadStocks());

async function init() {
  const [stocksResponse, metaResponse] = await Promise.all([
    fetch("./data/stocks.json"),
    fetch("./data/meta.json"),
  ]);
  state.allStocks = await stocksResponse.json();
  const meta = await metaResponse.json();
  state.sectors = meta.sectors || [];
  elements.refreshStamp.textContent = new Date(meta.generated_at).toLocaleString();
  elements.dataSource.textContent = meta.source || "unknown";
  elements.refreshSchedule.textContent = meta.refresh_schedule || "manual";
  await loadStocks();
}

init();
