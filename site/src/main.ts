import './style.css';
import type { Benchmark, HistoryPoint, LeverageGame, Summary } from './types';

const LABELS: Record<string, string> = {
  '24_wins': '24 regular-season wins',
  top4: 'Top 4 in Big Ten play',
  btt_title: 'Win the Big Ten Tournament',
  sweet16: 'Reach the Sweet 16',
};

const pct = (p: number) => (p > 0.995 ? '>99%' : p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`);
const esc = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);

function normalizeBenchmarks(b: Summary['benchmarks']): Benchmark[] {
  if (Array.isArray(b)) return b;
  return Object.entries(b).map(([key, v]) =>
    typeof v === 'number' ? { key, p: v } : { key, p: v.p, label: v.label, status: v.status },
  );
}

async function load<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(path, { cache: 'no-store' });
    return res.ok ? ((await res.json()) as T) : null;
  } catch {
    return null;
  }
}

function sparkline(points: HistoryPoint[]): string {
  if (points.length < 2) return '<p class="muted">The trend line appears after a few updates.</p>';
  const w = 640, h = 160, pad = 24;
  const top = Math.min(1, Math.ceil((Math.max(...points.map((pt) => pt.p_extension)) * 1.4) * 10) / 10);
  const x = (i: number) => pad + (i / (points.length - 1)) * (w - 2 * pad);
  const y = (p: number) => h - pad - (p / top) * (h - 2 * pad);
  const d = points.map((pt, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(pt.p_extension).toFixed(1)}`).join('');
  const last = points.at(-1)!;
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Extension odds over time">
    <line x1="${pad}" x2="${w - pad}" y1="${y(top)}" y2="${y(top)}" class="grid"/>
    <text x="${pad}" y="${y(top) - 6}" class="axis">${Math.round(top * 100)}%</text>
    <line x1="${pad}" x2="${w - pad}" y1="${y(0)}" y2="${y(0)}" class="grid"/>
    <path d="${d}" class="line"/>
    <circle cx="${x(points.length - 1)}" cy="${y(last.p_extension)}" r="4" class="dot"/>
    <text x="${pad}" y="${h - 4}" class="axis">${esc(points[0].date)}</text>
    <text x="${w - pad}" y="${h - 4}" class="axis" text-anchor="end">${esc(last.date)}</text>
  </svg>`;
}

function leverageRows(games: LeverageGame[]): string {
  if (!games.length) return '<p class="muted">No remaining games.</p>';
  // How much a game is expected to move the odds: swing times the outcome's spread.
  // A 99% favourite has a big swing but almost never loses, so it rarely matters.
  const impact = (g: LeverageGame) => (g.p_ext_if_win - g.p_ext_if_loss) * Math.sqrt(g.p_win * (1 - g.p_win));
  const maxImpact = Math.max(...games.map(impact));
  const rows = games
    .map((g) => {
      const swing = g.p_ext_if_win - g.p_ext_if_loss;
      return `<tr class="${impact(g) === maxImpact ? 'hot' : ''}">
        <td>${esc(g.date)}</td>
        <td>${g.site === 'A' ? '@ ' : g.site === 'N' ? 'vs ' : ''}${esc(g.opponent)}</td>
        <td class="num">${pct(g.p_win)}</td>
        <td class="num">${pct(g.p_ext_if_win)}</td>
        <td class="num">${pct(g.p_ext_if_loss)}</td>
        <td class="num strong">${(swing * 100).toFixed(1)} pts</td>
      </tr>`;
    })
    .join('');
  return `<div class="table-wrap"><table>
    <thead><tr><th>Date</th><th>Opponent</th><th class="num">Win chance</th><th class="num">Odds if win</th><th class="num">Odds if loss</th><th class="num">Swing</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function upcoming(games: LeverageGame[]): LeverageGame[] {
  const today = new Date().toISOString().slice(0, 10);
  return [...games]
    .filter((g) => g.date >= today)
    .sort((a, b) => a.date.localeCompare(b.date))
    .slice(0, 8);
}

async function main() {
  const app = document.getElementById('app')!;
  const [summary, history] = await Promise.all([
    load<Summary>('data/summary.json'),
    load<HistoryPoint[]>('data/history.json'),
  ]);
  if (!summary) {
    app.innerHTML = '<p class="loading">No simulation output yet. Check back after the next update.</p>';
    return;
  }
  const benches = normalizeBenchmarks(summary.benchmarks);
  const hist = history ?? [];
  const prev = hist.length >= 2 ? hist.at(-2)!.p_extension : null;
  const delta = prev === null ? '' : summary.p_extension - prev;

  app.innerHTML = `
    ${summary.as_of === 'illustrative' ? '<p class="banner">Illustrative ratings: these numbers come from sample data, not real ratings.</p>' : ''}
    <header>
      <p class="eyebrow">Michigan basketball</p>
      <h1>Will Mike Boynton earn his extension?</h1>
      <p class="lede">He gets an automatic extension by hitting 2 of these 4 benchmarks.</p>
    </header>
    <section class="hero">
      <div class="big">${pct(summary.p_extension)}</div>
      <div class="hero-meta">
        <div>chance he hits at least 2 of 4</div>
        ${typeof delta === 'number' ? `<div class="${delta >= 0 ? 'up' : 'down'}">${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)} pts since last update</div>` : ''}
        ${summary.record ? `<div class="muted">Record: ${summary.record.wins}-${summary.record.losses}</div>` : ''}
        ${summary.reg_wins_mean ? `<div class="muted">Projected regular-season wins: ${summary.reg_wins_mean.toFixed(1)}</div>` : ''}
      </div>
    </section>
    <section class="cards">
      ${benches
        .map(
          (b) => `<article class="card">
            <h2>${esc(b.label ?? LABELS[b.key] ?? b.key)}</h2>
            <div class="card-p">${pct(b.p)}</div>
            <div class="bar"><span style="width:${(b.p * 100).toFixed(1)}%"></span></div>
            ${b.status ? `<p class="muted">${esc(b.status)}</p>` : ''}
          </article>`,
        )
        .join('')}
    </section>
    <section>
      <h2>Odds over the season</h2>
      ${sparkline(hist)}
    </section>
    <section>
      <h2>Upcoming games</h2>
      <p class="muted">Michigan's next games, with the extension odds after a win or a loss. Swing is the gap between them. The highlighted game is the one expected to move the odds most, counting both its swing and how uncertain the result is.</p>
      ${leverageRows(upcoming(summary.leverage ?? []))}
    </section>
    <footer>
      <p>Method: we simulate the rest of the season, the Big Ten Tournament and the NCAA tournament thousands of times from team efficiency ratings, and count how often Boynton hits at least two benchmarks.
      Ratings via <a href="https://kenpom.com">KenPom</a>${summary.as_of && summary.as_of !== 'illustrative' ? `, as of ${esc(summary.as_of)}` : ''}. Not affiliated with the University of Michigan or KenPom.</p>
      ${summary.generated_at ? `<p class="muted">Updated ${esc(summary.generated_at)}</p>` : ''}
    </footer>`;
}

main();
