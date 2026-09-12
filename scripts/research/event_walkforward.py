"""Calendar/event-label expanding splits; not fitting, CPCV or strategy returns."""
import pandas as pd


def _utc(value):
    try:
        at = pd.Timestamp(value)
        if pd.isna(at) or at.tzinfo is None:
            raise ValueError('timezone-aware timestamp required')
        return at.tz_convert('UTC')
    except (TypeError, OverflowError) as exc:
        raise ValueError('invalid timestamp') from exc


def split_events(events, *, test_start, test_end, gap_minutes=0):
    """Full conservative label-end is inclusive; equality is purged/censored.

    Caller supplies maximum policy outcome horizon, NOT realized early exit.
    Fit examples/prompts/transforms only on returned training IDs. This helper
    cannot certify supplied horizons or historical knowledge contamination.
    """
    start, end = _utc(test_start), _utc(test_end)
    if start >= end or type(gap_minutes) is not int or gap_minutes < 0:
        raise ValueError('ordered window and nonnegative integer gap required')
    try:
        train_end = start - pd.Timedelta(minutes=gap_minutes)
    except (ValueError, OverflowError) as exc:
        raise ValueError('unrepresentable gap') from exc
    if not isinstance(events, list):
        raise ValueError('event list required')
    rows, seen = [], set()
    for event in events:
        if not isinstance(event, dict) or set(event) != {'id', 'decision_time', 'label_end'}:
            raise ValueError('exact event fields required')
        identity = event['id']
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError('unique nonempty event IDs required')
        seen.add(identity)
        decision, label_end = _utc(event['decision_time']), _utc(event['label_end'])
        if label_end < decision:
            raise ValueError('label ends before decision')
        rows.append((decision, identity, label_end))
    train, test, excluded = [], [], {}
    for decision, identity, label_end in sorted(rows):
        if decision < start:
            if label_end < train_end:
                train.append(identity)
            else:
                excluded[identity] = 'training_label_overlap_or_gap'
        elif decision < end:
            if label_end < end:
                test.append(identity)
            else:
                excluded[identity] = 'test_label_outside_window'
        else:
            excluded[identity] = 'outside_window'
    return dict(train_ids=train, test_ids=test, excluded=excluded,
                test_start=start.isoformat(), test_end=end.isoformat(),
                training_labels_before=train_end.isoformat(),
                kind='expanding_event_label_purged_walk_forward', fitted=False)
