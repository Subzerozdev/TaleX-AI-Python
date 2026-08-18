"""Unit tests for the DETERMINISTIC pure-logic helpers inside watermark_service.py.

watermark_service.py phần lớn là orchestration ffmpeg/OpenCV/pytesseract thật (embed_image_
watermark, embed_video_audio_watermark, embed_ab_watermark_hls, extract_ab_watermark_hls,
extract_image_watermark) — mock hết các lệnh subprocess/cv2 cho những hàm đó sẽ chỉ verify
"có gọi subprocess" chứ không verify đúng kết quả xử lý tín hiệu/hình ảnh thật, giá trị test
gần như bằng 0. CỐ Ý KHÔNG test các hàm đó ở đây.

Chỉ test các hàm con thuần túy/deterministic, không cần ffmpeg/OpenCV:
- _pad_id_v2 / _string_to_binary / _binary_to_string: string logic thuần
- _generate_ultrasound_wav + _extract_ook_from_audio: encode/decode OOK roundtrip THẬT
  (dùng đúng hàm production, không mock) — chỉ cần scipy.io.wavfile (đã có sẵn dependency),
  không cần ffmpeg.
"""
import numpy as np
import pytest
from scipy.io import wavfile

from app.services import watermark_service


class TestPadIdV2:
    def test_pads_short_id_to_128_chars(self):
        result = watermark_service._pad_id_v2("creator-abc", "viewer-xyz")
        assert len(result) == watermark_service.WM_SHAPE_LENGTH_NEW
        assert result.startswith("C:creator-abc|V:viewer-xyz")

    def test_defaults_viewer_id_to_none_marker(self):
        result = watermark_service._pad_id_v2("creator-abc")
        assert result.startswith("C:creator-abc|V:NONE")

    def test_truncates_id_longer_than_128_chars(self):
        long_creator_id = "x" * 200
        result = watermark_service._pad_id_v2(long_creator_id)
        assert len(result) == watermark_service.WM_SHAPE_LENGTH_NEW

    def test_strips_non_ascii_characters(self):
        # encode('ascii', 'ignore') loại bỏ ký tự non-ASCII thay vì throw.
        result = watermark_service._pad_id_v2("crëator-üñïcödé")
        assert "ë" not in result and "ü" not in result


class TestBinaryStringRoundtrip:
    def test_string_to_binary_produces_8_bits_per_char(self):
        binary = watermark_service._string_to_binary("AB")
        assert len(binary) == 16
        assert binary == format(ord("A"), "08b") + format(ord("B"), "08b")

    def test_roundtrip_recovers_original_string(self):
        original = "hello123-UUID"
        recovered = watermark_service._binary_to_string(watermark_service._string_to_binary(original))
        assert recovered == original

    def test_binary_to_string_ignores_incomplete_trailing_byte(self):
        # Chuỗi bit không chia hết cho 8 — byte cuối cùng thiếu bit bị bỏ qua (xem code:
        # `if len(byte) == 8`), không throw.
        binary = watermark_service._string_to_binary("A") + "101"
        result = watermark_service._binary_to_string(binary)
        assert result == "A"


class TestOokAudioRoundtrip:
    """Test thật encode (_generate_ultrasound_wav) → decode (_extract_ook_from_audio),
    dùng đúng hàm production, không mock — chỉ cần scipy, không cần ffmpeg."""

    def test_decodes_short_id_correctly(self, tmp_path):
        creator_id = "abc123"
        wav_path = str(tmp_path / "ultrasound.wav")

        watermark_service._generate_ultrasound_wav(creator_id, wav_path)
        sample_rate, audio_data = wavfile.read(wav_path)

        extracted = watermark_service._extract_ook_from_audio(
            audio_data.astype(np.float64), sample_rate, freq_target=8000.0, bit_duration=0.05
        )

        assert extracted == creator_id

    def test_decodes_uuid_like_id_correctly(self, tmp_path):
        creator_id = "550e8400-e29b-41d4-a716"
        wav_path = str(tmp_path / "ultrasound.wav")

        watermark_service._generate_ultrasound_wav(creator_id, wav_path)
        sample_rate, audio_data = wavfile.read(wav_path)

        extracted = watermark_service._extract_ook_from_audio(
            audio_data.astype(np.float64), sample_rate, freq_target=8000.0, bit_duration=0.05
        )

        assert extracted == creator_id

    def test_returns_none_for_pure_silence(self):
        # Không có header "10101010" nào trong tín hiệu toàn 0 — phải trả None, không throw
        # và không trả bừa 1 chuỗi rác.
        sample_rate = 44100
        silence = np.zeros(sample_rate * 2, dtype=np.float64)

        result = watermark_service._extract_ook_from_audio(
            silence, sample_rate, freq_target=8000.0, bit_duration=0.05
        )

        assert result is None

    def test_returns_none_for_random_noise_without_header(self):
        # Nhiễu ngẫu nhiên không chứa header hợp lệ — không được nhận nhầm là watermark thật.
        rng = np.random.default_rng(seed=42)
        sample_rate = 44100
        noise = rng.normal(0, 100, sample_rate * 2)

        result = watermark_service._extract_ook_from_audio(
            noise, sample_rate, freq_target=8000.0, bit_duration=0.05
        )

        # Nhiễu ngẫu nhiên CÓ THỂ tình cờ khớp header ngắn (8 bit) nhưng payload sau đó gần
        # như chắc chắn không đạt tỉ lệ alphanum >= 0.6 → vẫn phải trả None.
        assert result is None
