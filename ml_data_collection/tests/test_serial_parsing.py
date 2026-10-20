"""
test_serial_parsing.py
======================
Unit tests for verifying serial record parsing against the exact 9-column CSV schema:
timestamp,vib_rms,vib_peak,vib_x,vib_y,vib_z,current,temperature,label
"""

from pathlib import Path
import sys

scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from serial_logger import parse_and_validate_record


def test_user_example_rows():
    examples = [
        ("14523,0.9821,1.2340,0.3210,0.8120,0.9910,0.4821,32.50,0", 0, 0.9821, 0.4821, 32.50),
        ("15025,0.9934,1.1980,0.3190,0.8240,0.9870,0.4799,32.60,0", 0, 0.9934, 0.4799, 32.60),
        ("15528,1.8432,2.3210,0.9120,1.4320,1.8100,0.3201,33.20,1", 1, 1.8432, 0.3201, 33.20),
        ("16031,1.9120,2.4100,0.9340,1.5010,1.8920,0.3180,33.50,1", 1, 1.9120, 0.3180, 33.50),
        ("16534,0.9810,1.2100,0.3100,0.8010,0.9800,0.7821,34.10,2", 2, 0.9810, 0.7821, 34.10),
    ]

    for line, exp_label, exp_rms, exp_curr, exp_temp in examples:
        res = parse_and_validate_record(line)
        assert res is not None, f"Failed to parse line: {line}"
        assert res["label"] == exp_label
        assert abs(res["vib_rms"] - exp_rms) < 1e-4
        assert abs(res["current"] - exp_curr) < 1e-4
        assert abs(res["temperature"] - exp_temp) < 1e-2


def test_header_and_comments():
    assert parse_and_validate_record("timestamp,vib_rms,vib_peak,vib_x,vib_y,vib_z,current,temperature,label") is None
    assert parse_and_validate_record("# SESSION COMPLETE - Label: 1") is None
    assert parse_and_validate_record("# Starting 3-minute logging session") is None


def test_malformed_rows():
    # Out of bound vibration
    assert parse_and_validate_record("1000,55.0,60.0,0.1,0.2,0.3,0.5,30.0,0") is None
    # Out of bound temperature
    assert parse_and_validate_record("1000,1.0,1.2,0.1,0.2,0.3,0.5,200.0,0") is None
    # Out of bound label
    assert parse_and_validate_record("1000,1.0,1.2,0.1,0.2,0.3,0.5,30.0,9") is None
    # Truncated
    assert parse_and_validate_record("1000,1.0,1.2,0.1") is None
    # Non-numeric
    assert parse_and_validate_record("abc,def,ghi,jkl,mno,pqr,stu,vwx,yz") is None


if __name__ == "__main__":
    test_user_example_rows()
    test_header_and_comments()
    test_malformed_rows()
    print("All 9-column serial parser unit tests passed successfully!")
