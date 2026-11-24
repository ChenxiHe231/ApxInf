# HF container decoding and chat templating. Model computation stays native CUDA.
import base64
import io
import http.client
import ipaddress
import json
import signal
import socket
import struct
import sys
import urllib.parse
import urllib.request

MAX_BYTES = 8 * 1024 * 1024


def public_https(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("image URLs require public HTTPS without credentials")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("image URLs must resolve to public addresses")
    return url


class PublicHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        # Connect to the address that was validated, retaining hostname/SNI for TLS.
        addresses = socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise ValueError("image URLs must resolve to public addresses")
        address = addresses[0][4][:2]
        # A deployment may require an administrator-configured HTTP proxy. Pin
        # the CONNECT target to the validated IP as well, so the proxy cannot
        # resolve the user-supplied hostname to a different/private address.
        proxy = urllib.request.getproxies().get("https")
        if proxy and not urllib.request.proxy_bypass(self.host):
            parsed = urllib.parse.urlsplit(proxy)
            if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("image downloads support HTTP proxies without credentials")
            self.set_tunnel(address[0], address[1])
            self.sock = socket.create_connection((parsed.hostname, parsed.port or 80), self.timeout)
            self._tunnel()
        else:
            self.sock = socket.create_connection(address, self.timeout)
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


class PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(PublicHTTPSConnection, request, context=self._context)


class Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        return super().redirect_request(request, fp, code, message, headers, public_https(url))


def load_image(url):
    from PIL import Image, ImageOps
    if url.startswith("data:image/"):
        header, encoded = url.split(",", 1)
        if not header.endswith(";base64") or len(encoded) > (MAX_BYTES * 4 // 3 + 4):
            raise ValueError("invalid or oversized image data URL")
        data = base64.b64decode(encoded, validate=True)
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), PublicHTTPSHandler(), Redirects())
        with opener.open(public_https(url), timeout=15) as response:
            data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("encoded image exceeds 8 MiB")
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 16777216 or getattr(image, "n_frames", 1) != 1:
            raise ValueError("only static images up to 16 megapixels are supported")
        return ImageOps.exif_transpose(image).convert("RGB")


def process(request, processor=None):
    import numpy as np
    from transformers import AutoProcessor
    messages = request["messages"]
    image_count = 0
    for message in messages:
        if not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if part["type"] == "image_url":
                image_count += 1
                if image_count > 16:
                    raise ValueError("at most 16 images are supported per request")
                image = load_image(part["image_url"]["url"])
                part.clear()
                part.update(type="image", image=image)
    if image_count == 0:
        raise ValueError("image request has no images")
    if processor is None:
        processor = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
    # Bound patch count and tensor transfer while preserving processor resize semantics.
    patch_size = processor.image_processor.patch_size
    processor.image_processor.size = {"shortest_edge": 65536,
                                      "longest_edge": min(1048576, 4096 * patch_size * patch_size // image_count)}
    options = request["options"]
    if {"messages", "tokenize", "return_dict", "return_tensors", "add_generation_prompt"} & options.keys():
        raise ValueError("reserved processor template option")
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt", **options)
    pixels = inputs["pixel_values"].cpu().numpy().astype("<f4")
    metadata = json.dumps({"tokens": inputs["input_ids"][0].tolist(), "shape": list(pixels.shape),
                           "grids": inputs["image_grid_thw"].tolist()}).encode()
    return metadata, pixels.tobytes()


def write_frame(metadata, pixels=b""):
    sys.stdout.buffer.write(struct.pack("<I", len(metadata)))
    sys.stdout.buffer.write(metadata)
    sys.stdout.buffer.write(pixels)
    sys.stdout.buffer.flush()


def error_frame(error):
    unavailable = isinstance(error, (ImportError, ModuleNotFoundError))
    message = str(error) if not unavailable else "configure APXINF_PYTHON with the image preprocessing dependencies"
    return json.dumps({"error": message, "unavailable": unavailable}).encode()


def worker():
    # Startup and every request have a watchdog; Rust owns/reaps this process.
    signal.alarm(90)
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
    except Exception as error:
        write_frame(error_frame(error))
        return
    write_frame(b'{"ready":true}')
    signal.alarm(0)
    while True:
        length = sys.stdin.buffer.read(4)
        if not length:
            return
        signal.alarm(90)
        if len(length) != 4:
            raise ValueError("truncated image processor request header")
        count = struct.unpack("<I", length)[0]
        if count > 32 * 1024 * 1024:
            raise ValueError("oversized image processor request")
        data = sys.stdin.buffer.read(count)
        if len(data) != count:
            raise ValueError("truncated image processor request")
        try:
            write_frame(*process(json.loads(data), processor))
        except Exception as error:
            write_frame(error_frame(error))
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    if "--worker" in sys.argv[2:]:
        worker()
    else:
        signal.alarm(90)
        write_frame(*process(json.load(sys.stdin)))
