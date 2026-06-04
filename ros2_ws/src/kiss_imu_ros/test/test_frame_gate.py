from kiss_imu_ros.frame_gate import FrameGate


def test_enter_then_busy_then_release():
    gate = FrameGate()
    assert gate.try_enter() is True       # first entry succeeds
    assert gate.try_enter() is False      # busy -> dropped
    assert gate.dropped == 1
    assert gate.try_enter() is False      # still busy -> dropped again
    assert gate.dropped == 2
    gate.exit()
    assert gate.try_enter() is True       # free again
    assert gate.dropped == 2              # successful entry does not increment
    gate.exit()


def test_dropped_starts_zero():
    assert FrameGate().dropped == 0
