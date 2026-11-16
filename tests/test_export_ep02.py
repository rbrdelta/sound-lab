"""export_ep02.py: the loudness-alone check, on made-up clip rows (no real clip or reading is opened)."""
import export_ep02 as ex


def rows(levels):
    return [{"label": lab, "rms_db": str(db)} for lab, dbs in levels.items() for db in dbs]


def test_loudness_alone_gets_every_clip_right_when_groups_do_not_overlap():
    r = rows({"happy": [-20, -21, -19], "sad": [-30, -31, -29]})
    assert ex.loudness_alone(r, ("happy", "sad")) == 6


def test_loudness_alone_leaves_the_clip_out_of_its_own_group_average():
    # Left out, the -24 happy clip is nearer the sad average (-27, 3 away) than the other happy
    # clips' (-20, 4 away): a miss. Counted in its own group, the happy average would be -21.3
    # (2.7 away) and it would wrongly count as right, giving 6.
    r = rows({"happy": [-20, -20, -24], "sad": [-27, -27, -27]})
    assert ex.loudness_alone(r, ("happy", "sad")) == 5


def test_loudness_alone_ignores_clips_outside_the_check():
    r = rows({"happy": [-20, -21], "sad": [-30, -31], "neutral": [-25, -25]})
    assert ex.loudness_alone(r, ("happy", "sad")) == 4
