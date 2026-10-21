"""FigureWriter — the article's four charts, as SVG and PNG, from tables the notebook already has."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from modgud.app import implicit_return

from rapsheet.core import Frame, PathLike

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#8a8984'
HAIRLINE = '#e6e5e1'
# Validated categorical order (blue, orange): adjacent CVD ΔE 24.7, normal-vision ΔE 33.6, ≥ 3:1.
SERIES = ('#2a78d6', '#eb6834')
LABEL = {'color': INK_SECONDARY, 'fontsize': 9.5}
NOTE = {'color': INK_SECONDARY, 'fontsize': 8.5}


@dataclass(frozen=True)
class FigureWriter:
  """Draw F1–F4 in one quiet house style and save each as ``<stem>.svg`` and ``<stem>.png``.

  Inputs are plain tables from ``rapsheet.evaluation`` (reliability bins, risk–coverage curves,
  the confusion ranking, the budget sweep), so this package never touches a model or a logit.
  Marks follow the dataviz rules: 2px lines, ringed markers, hairline grids, text in ink colors
  (never the series color), and a legend whenever there are two series.
  """

  directory: Path
  dpi: int = 200
  width: float = 8.0

  @classmethod
  @implicit_return
  def at(cls, directory: PathLike) -> 'FigureWriter':
    """Return a writer that saves into ``directory`` (created if needed)."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    cls(directory=target)

  # ------------------------------------------------------------------ shared

  @staticmethod
  @implicit_return
  def percent(value: float) -> str:
    """Return a compact percentage: 0.005 → '0.5%', 0.01 → '1%'."""
    f'{value * 100:g}%'

  @implicit_return
  def styled(self, ax: Any) -> Any:
    """Apply the surface, a hairline grid and quiet spines to one axes."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=HAIRLINE, linewidth=0.8, linestyle='-')
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    for side in ('top', 'right', 'left'):
      ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(HAIRLINE)
    ax

  @implicit_return
  def figure(self, height: float, title: str) -> Figure:
    """Return an empty figure on the surface color with a left-aligned bold title."""
    figure = Figure(figsize=(self.width, height), facecolor=SURFACE, layout='constrained')
    figure.suptitle(title, x=0.01, ha='left', fontsize=13, fontweight='bold', color=INK)
    figure

  def legend(self, ax: Any) -> None:
    """Add an unboxed legend in secondary ink."""
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc='best')

  @implicit_return
  def save(self, figure: Figure, stem: str) -> Path:
    """Write SVG and PNG; return the PNG path."""
    for suffix in ('svg', 'png'):
      figure.savefig(self.directory / f'{stem}.{suffix}', dpi=self.dpi, facecolor=SURFACE)
    self.directory / f'{stem}.png'

  # ------------------------------------------------------------------ F1

  @implicit_return
  def reliability(self, tables: dict[str, Frame], subtitle: str) -> Path:
    """F1: per-bin accuracy against confidence, one line per split, plus rows per bin below."""
    figure = self.figure(5.8, 'F1 · Does 90% confident mean 90% right?')
    grid = figure.add_gridspec(2, 1, height_ratios=(3.2, 1))
    top = self.styled(figure.add_subplot(grid[0]))
    bottom = self.styled(figure.add_subplot(grid[1], sharex=top))
    top.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, label='perfectly calibrated')
    for i, (name, table) in enumerate(tables.items()):
      color = SERIES[i % len(SERIES)]
      top.plot(
        table['confidence'],
        table['accuracy'],
        color=color,
        linewidth=2,
        marker='o',
        markersize=6.5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        solid_joinstyle='round',
        label=name,
      )
      span = table['high'] - table['low']
      bar = span * 0.8 / len(tables)
      bottom.bar(
        table['low'] + span * 0.1 + (i + 0.5) * bar,
        table['rows'] / table['rows'].sum(),
        width=bar,
        color=color,
      )
    top.set_xlim(0, 1.0)
    top.set_ylim(0, 1.02)
    top.set_ylabel('accuracy in bin', **LABEL)
    top.tick_params(labelbottom=False)
    top.set_title(subtitle, loc='left', **LABEL)
    bottom.set_ylabel('share of rows', **LABEL)
    bottom.set_xlabel('model confidence (top-class probability)', **LABEL)
    bottom.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    self.legend(top)
    self.save(figure, 'f1-reliability')

  # ------------------------------------------------------------------ F2

  @implicit_return
  def risk_coverage(
    self,
    curves: dict[str, Frame],
    points: dict[str, tuple[float, float]],
    budget: float,
    subtitle: str,
  ) -> Path:
    """F2: selective risk against coverage per split, the budget line, and the policy's points."""
    figure = self.figure(4.8, 'F2 · Risk against coverage')
    ax = self.styled(figure.add_subplot())
    highest = max((risk for _, risk in points.values()), default=0.0)  # labels stack below it
    for i, (name, curve) in enumerate(curves.items()):
      color = SERIES[i % len(SERIES)]
      ax.plot(curve['coverage'], curve['risk'], color=color, linewidth=2, label=name)
      if name in points:
        coverage, risk = points[name]
        ax.plot(
          [coverage],
          [risk],
          marker='o',
          markersize=8,
          color=color,
          markeredgecolor=SURFACE,
          markeredgewidth=2,
          linestyle='none',
        )
        ax.annotate(
          f'{name}: per-class policy decides {coverage:.0%} alone, {risk:.1%} wrong',
          (coverage, risk),
          xytext=(coverage + 0.04, highest - 0.0035 * (len(points) - 1 - i)),
          va='center',
          arrowprops={'arrowstyle': '-', 'color': INK_MUTED, 'linewidth': 0.8, 'shrinkB': 5},
          **NOTE,
        )
    ax.axhline(budget, color=INK_MUTED, linewidth=1)
    ax.annotate(
      f'error budget {self.percent(budget)}',
      (0.98, budget),
      xytext=(0, 4),
      textcoords='offset points',
      ha='right',
      **NOTE,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(float(c['risk'].max()) for c in curves.values()) * 1.1)
    ax.xaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:.1%}')
    ax.set_xlabel('coverage (share of rows the model decides alone, most confident first)', **LABEL)
    ax.set_ylabel('error rate among those rows', **LABEL)
    ax.set_title(subtitle, loc='left', **LABEL)
    self.legend(ax)
    self.save(figure, 'f2-risk-coverage')

  # ------------------------------------------------------------------ F3

  @implicit_return
  def top_confusions(self, table: Frame, subtitle: str) -> Path:
    """F3: the most common mistakes as horizontal bars, labelled true → predicted."""
    ordered = table.iloc[::-1].reset_index(drop=True)
    figure = self.figure(0.4 * len(ordered) + 1.6, 'F3 · Where the model is wrong')
    ax = self.styled(figure.add_subplot())
    names = [f'{t}  →  {p}' for t, p in zip(ordered['true'], ordered['predicted'], strict=True)]
    ax.barh(range(len(ordered)), ordered['rows'], height=0.5, color=SERIES[0])
    for y, rows in enumerate(ordered['rows']):
      ax.annotate(
        f'{rows:,}', (rows, y), xytext=(4, 0), textcoords='offset points', va='center', **NOTE
      )
    ax.set_yticks(range(len(ordered)), names, fontsize=8.5, color=INK)
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, float(ordered['rows'].max()) * 1.12)
    ax.set_xlabel('rows (true label → predicted label)', **LABEL)
    ax.set_title(subtitle, loc='left', **LABEL)
    self.save(figure, 'f3-top-confusions')

  # ------------------------------------------------------------------ F4

  @implicit_return
  def budget_sweep(self, sweep: Frame, splits: tuple[str, ...], subtitle: str) -> Path:
    """F4: per split, coverage (top) and realized error (bottom) against the error budget."""
    figure = self.figure(6.8, 'F4 · What each error budget buys, and what it costs')
    grid = figure.subplots(2, len(splits), sharex=True, sharey='row', squeeze=False)
    policies = (('per_class', 'per-class thresholds'), ('global', 'one global threshold'))
    for column, split in enumerate(splits):
      part = sweep.xs(split, level='split')
      top, bottom = self.styled(grid[0][column]), self.styled(grid[1][column])
      for i, (policy, label) in enumerate(policies):
        for ax, measure in ((top, 'coverage'), (bottom, 'error')):
          ax.plot(
            part.index,
            part[f'{policy}_{measure}'],
            color=SERIES[i],
            linewidth=2,
            marker='o',
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=label,
          )
      bottom.plot(part.index, part.index, color=INK_MUTED, linewidth=1, label='the budget itself')
      top.set_title(split, loc='left', fontsize=10, color=INK)
      top.set_ylim(0, 1.02)
      top.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
      bottom.set_xscale('log')
      bottom.set_xticks(list(part.index), [self.percent(b) for b in part.index], fontsize=8)
      bottom.minorticks_off()
      bottom.set_ylim(0, None)
      bottom.yaxis.set_major_formatter(lambda v, _: f'{v:.1%}')
      bottom.set_xlabel('error budget (log scale)', **LABEL)
    grid[0][0].set_ylabel('share of rows decided alone', **LABEL)
    grid[1][0].set_ylabel('error among those rows', **LABEL)
    handles, labels = grid[1][0].get_legend_handles_labels()
    grid[0][-1].legend(
      handles, labels, loc='lower right', frameon=False, fontsize=9, labelcolor=INK_SECONDARY
    )
    figure.supxlabel(subtitle, x=0.01, ha='left', **NOTE)
    self.save(figure, 'f4-budget-sweep')
