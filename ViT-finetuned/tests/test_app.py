import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import vit_runtime as runtime


def png(mode='RGB'):
    data = io.BytesIO()
    Image.new(mode, (32, 32)).save(data, format='PNG')
    return data.getvalue()


class ImageValidationTests(unittest.TestCase):
    def test_invalid_and_oversized_uploads(self):
        for data in [b'', b'not an image', b'x' * (runtime.MAX_UPLOAD_BYTES + 1)]:
            with self.assertRaises(ValueError):
                runtime.read_image(data)
        with patch.object(runtime, 'MAX_PIXELS', 100):
            with self.assertRaises(ValueError):
                runtime.read_image(png())

    def test_rgb_conversion(self):
        for mode in ['L', 'RGBA', 'RGB']:
            self.assertEqual(runtime.read_image(png(mode)).mode, 'RGB')

    def test_initial_and_invalid_ui(self):
        app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        with patch('streamlit.file_uploader', return_value=io.BytesIO(b'bad')):
            app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn('could not be read', app.warning[0].value)

    def test_failed_load_hides_details_and_offers_retry(self):
        import streamlit as st
        st.cache_resource.clear()
        with patch('streamlit.file_uploader', return_value=io.BytesIO(png())), \
             patch.object(runtime, 'load_model', side_effect=RuntimeError('SECRET_TEST_TOKEN private path')):
            app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn('temporarily unavailable', app.error[0].value)
        self.assertEqual(app.button[0].label, 'Retry classification')
        self.assertNotIn('SECRET_TEST_TOKEN', str(app))

    def test_manual_retry_recovers_with_same_upload(self):
        import streamlit as st
        st.cache_resource.clear()
        data = io.BytesIO(png())
        rows = [{'id': 6, 'label': 'beignets', 'probability': .75}]
        with patch('streamlit.file_uploader', return_value=data), \
             patch.object(runtime, 'load_model', side_effect=[RuntimeError('temporary'), (object(), object())]), \
             patch.object(runtime, 'predict', return_value=rows):
            app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
            self.assertEqual(app.button[0].label, 'Retry classification')
            app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any('beignets' in item.value and '75.00%' in item.value for item in app.markdown))


if __name__ == '__main__':
    unittest.main()
