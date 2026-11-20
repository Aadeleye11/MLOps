import io
import unittest

from PIL import Image, ImageDraw

from scripts.audit_similar_images import Photo, difference_hash, find_pairs


class SimilarityTest(unittest.TestCase):
    def test_jpeg_recompression(self):
        image = Image.new("RGB", (160, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((20, 10, 130, 90), fill="green")
        draw.line((5, 110, 122, 18), fill="black", width=3)
        encodings = []
        hashes = []
        for quality in (95, 45):
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=quality)
            encodings.append(encoded.getvalue())
            with Image.open(io.BytesIO(encoded.getvalue())) as decoded:
                hashes.append(difference_hash(decoded))
        self.assertNotEqual(*encodings)
        self.assertLessEqual((hashes[0] ^ hashes[1]).bit_count(), 4)

    def test_index_matches_exhaustive_search_at_distance_boundary(self):
        base = 0xAF019123789BCD08
        left = [Photo("a", "rice", b"a", base), Photo("z", "maize", b"z", 0)]
        masks = [0, 1, sum(1 << n for n in (12, 25, 38, 63)),
                 sum(1 << n for n in (12, 25, 38, 51, 63))]
        right = [Photo(str(i), "maize", bytes([i]), base ^ mask)
                 for i, mask in enumerate(masks)]
        right.append(Photo("copy", "rice", b"a", base))
        expected = {(a.path, b.path) for a in left for b in right
                    if a.sha256 != b.sha256 and (a.dhash ^ b.dhash).bit_count() <= 4}
        pairs, exact = find_pairs(left, right)
        self.assertEqual(exact, 1)
        self.assertEqual(len(pairs), len(expected))
        self.assertEqual({(p["left"], p["right"]) for p in pairs}, expected)
