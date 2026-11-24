"""Run with the configured processor interpreter; no GPU or network required."""
import base64
import importlib.util
import io
from pathlib import Path
import unittest
import json
import struct
import sys
import types
from unittest.mock import patch, Mock
from PIL import Image

spec = importlib.util.spec_from_file_location("image_processor", Path(__file__).parents[1] / "src/image_processor.py")
processor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processor)


class ImageProcessorTest(unittest.TestCase):
    def test_decodes_a_static_image(self):
        stream = io.BytesIO()
        Image.new("RGB", (32, 48), "blue").save(stream, format="PNG")
        image = processor.load_image("data:image/png;base64," + base64.b64encode(stream.getvalue()).decode())
        self.assertEqual(image.size, (32, 48))
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))

    def test_honors_jpeg_orientation(self):
        stream = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (32, 48), "blue").save(stream, format="JPEG", exif=exif)
        image = processor.load_image("data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode())
        self.assertEqual(image.size, (48, 32))

    def test_rejects_bad_base64_and_animated_images(self):
        with self.assertRaises(ValueError):
            processor.load_image("data:image/png;base64,!!!")
        stream = io.BytesIO()
        Image.new("RGB", (32, 32), "red").save(stream, format="GIF", save_all=True,
            append_images=[Image.new("RGB", (32, 32), "blue")], duration=100, loop=0)
        with self.assertRaisesRegex(ValueError, "static"):
            processor.load_image("data:image/gif;base64," + base64.b64encode(stream.getvalue()).decode())

    def test_rejects_local_schemes_and_private_dns(self):
        for url in ["file:///etc/passwd", "http://example.com/a", "https://u:p@example.com/a"]:
            with self.assertRaises(ValueError):
                processor.public_https(url)
        for address in ["127.0.0.1", "192.168.1.1", "::1", "169.254.169.254"]:
            with patch.object(processor.socket, "getaddrinfo", return_value=[(0, 0, 0, "", (address, 443))]):
                with self.assertRaises(ValueError):
                    processor.public_https("https://example.com/a")

    def test_worker_reuses_processor_and_recovers_after_a_bad_request(self):
        fake = types.ModuleType("transformers")
        fake.AutoProcessor = Mock()
        loaded = fake.AutoProcessor.from_pretrained.return_value
        requests = [{"request": "first"}, {"request": "bad"}, {"request": "last"}]
        payload = b"".join(struct.pack("<I", len(raw)) + raw
                           for raw in (json.dumps(item).encode() for item in requests))
        stdin = types.SimpleNamespace(buffer=io.BytesIO(payload))
        stdout = types.SimpleNamespace(buffer=io.BytesIO())
        metadata = b'{"shape":[1,1]}'
        pixels = struct.pack("<f", 1.0)
        with patch.dict(sys.modules, {"transformers": fake}), \
             patch.object(processor.sys, "stdin", stdin), \
             patch.object(processor.sys, "stdout", stdout), \
             patch.object(processor.sys, "argv", ["processor", "/checkpoint", "--worker"]), \
             patch.object(processor.signal, "alarm"), \
             patch.object(processor, "process", side_effect=[(metadata, pixels), ValueError("bad image"), (metadata, pixels)]) as run:
            processor.worker()
        fake.AutoProcessor.from_pretrained.assert_called_once_with("/checkpoint", local_files_only=True)
        self.assertEqual([call.args for call in run.call_args_list], [(item, loaded) for item in requests])
        stream = io.BytesIO(stdout.buffer.getvalue())
        headers = []
        for _ in range(4):
            length = struct.unpack("<I", stream.read(4))[0]
            header = json.loads(stream.read(length))
            headers.append(header)
            if "shape" in header:
                self.assertEqual(stream.read(4), pixels)
        self.assertTrue(headers[0]["ready"])
        self.assertEqual(headers[2]["error"], "bad image")
        self.assertEqual(stream.read(), b"")

    def test_pins_the_validated_address_and_preserves_tls_hostname(self):
        connection = processor.PublicHTTPSConnection("example.com", timeout=5)
        with patch.object(processor.socket, "getaddrinfo", return_value=[(0, 0, 0, "", ("8.8.8.8", 443))]), \
             patch.object(processor.urllib.request, "getproxies", return_value={}), \
             patch.object(processor.socket, "create_connection") as create, \
             patch.object(connection._context, "wrap_socket") as wrap:
            connection.connect()
            create.assert_called_once_with(("8.8.8.8", 443), 5)
            wrap.assert_called_once_with(create.return_value, server_hostname="example.com")

    def test_proxy_connects_to_validated_ip_with_original_tls_hostname(self):
        connection = processor.PublicHTTPSConnection("example.com", timeout=5)
        with patch.object(processor.socket, "getaddrinfo", return_value=[(0, 0, 0, "", ("8.8.8.8", 443))]), \
             patch.object(processor.urllib.request, "getproxies", return_value={"https": "http://10.0.0.1:1089"}), \
             patch.object(processor.urllib.request, "proxy_bypass", return_value=False), \
             patch.object(processor.socket, "create_connection") as create, \
             patch.object(connection, "_tunnel") as tunnel, \
             patch.object(connection._context, "wrap_socket") as wrap:
            connection.connect()
            create.assert_called_once_with(("10.0.0.1", 1089), 5)
            self.assertEqual((connection._tunnel_host, connection._tunnel_port), ("8.8.8.8", 443))
            tunnel.assert_called_once_with()
            wrap.assert_called_once_with(create.return_value, server_hostname="example.com")


if __name__ == "__main__":
    unittest.main()
