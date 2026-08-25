"""
Stage 2.1 central switches for the correctness pass.

Every behavioural change in this stage is gated here so its isolated effect
can be measured (old vs. new on an unchanged data range) instead of being
baked in silently. Defaults are the new/intended behaviour; flip a field to
recover the pre-2.1 path for an A/B.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    # Entry-timing convention (2.1 entry refactor).
    #   True  (default): the 5m pullback/trigger read only *completed* candles
    #     (through bar i-1), the entry-stop ATR uses bar i-1, and the fill is
    #     bar i's OWN open. The whole decision is knowable at bar i's open, so
    #     nothing that waits for bar i to finish is used.
    #   False: pre-2.1 behaviour - the 5m read includes the current forming
    #     bar's close[i], the entry-stop ATR uses bar i, and the fill is bar
    #     i+1's open (the t -> t+1 stagger).
    entry_on_completed_bar: bool = True

    # Item 2 lag-curve DIAGNOSTIC only. Delays the entry fill by this many 5m
    # bars (0 = the real strategy). Used to check the entry edge decays
    # gradually with delay rather than collapsing between lag 0 and lag 1.
    entry_lag_bars: int = 0

    # Item 3.5 (A2): fill a stop / trailing-stop exit at the worse of the line
    # and the bar's open, so an overnight/news gap through the stop is not
    # filled at the (better) stop price the market never traded. Take-profit
    # stays a limit at its trigger. Default True = honest.
    gap_fill_exits: bool = True

    # Item 3 (A1): trigger stop / target / trailing on the bar's high/low
    # (intrabar), not only on the close. Default True = honest.
    range_based_exit_trigger: bool = True

    # Item 3 (both-hit): when a stop and target both fall inside one bar (only
    # possible under range_based_exit_trigger), resolve the stop first
    # (pessimistic). Default True = honest.
    pessimistic_both_hit: bool = True

    # A3: the chandelier trail for bar i is computed from the extreme through
    # i-1, and this bar's high/low are folded in only AFTER the trail has been
    # tested. Otherwise the level tested against bar i depends on bar i's own
    # high - which is knowable only if the high preceded the low, an intrabar
    # order the engine cannot know (G4). Default True = honest.
    trail_extreme_prev_bar: bool = True

    # A6: the risk ATR (entry stop distance AND chandelier trail) is read from
    # ATR(14) on 30m - the last fully closed 30m bar - instead of ATR(14) on 5m.
    # The strategy's thesis is a 2h-trend pullback and the stop's other leg (the
    # swing platform) is already measured on 30m, so a 5m ATR made one formula
    # compare two timeframes. Both legs move together: rr_trigger(2) x
    # sl_atr_mult(1.5) == chandelier_mult(3.0) puts the trail at breakeven when
    # the partial TP fires, which only holds if both use the same ATR.
    # Default True = coherent.
    risk_atr_on_30m: bool = True

    # A5: the cooldown is a 2h-timescale rule but is called once per 5m bar.
    # True = evaluate it once per NEW 2h bar (release takes 3 x 2h = 6 hours, as
    # the class docstring intends). False = pre-A5, counts calls (3 x 5m = 15
    # minutes, i.e. the cooldown barely engages). Default True = honest.
    cooldown_counts_2h_bars: bool = True

    # Item 2.6: no entries permitted before this many 5m bars (0 = no discard,
    # the default - indicators already return neutral until their window fills).
    # A non-zero explicit warm-up buffer (~5-10x the longest lookback: the 50-bar
    # 2h MA / 20-day vol) is a PARAMETER CHOICE left to the owner (§1.4), not set
    # here. See v2.1-LEDG-修复前后指标台账.md, Item 7.
    warmup_discard_bars: int = 0
