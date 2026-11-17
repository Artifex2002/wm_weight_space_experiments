import numpy as np
from pathlib import Path
from .packet import verify


def schema(name, description, properties):
    return {"type": "function", "name": name, "description": description,
            "strict": True, "parameters": {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}}


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
NUMBER = {"type": "number"}
VECTOR = {"type": "array", "items": NUMBER}
MATRIX = {"type": "array", "items": VECTOR}
TOOLS = [
    schema("list_tensors", "List checkpoint tensor names/shapes, paginated; no inference.",
           {"offset": INTEGER, "limit": INTEGER}),
    schema("read_tensor", "Read at most 1024 consecutive flattened weight values (C order).",
           {"name": STRING, "offset": INTEGER, "count": INTEGER}),
    schema("tensor_stats", "Weight-only min/max/mean/std/L2, without model execution.", {"name": STRING}),
    schema("matrix_spectrum", "Singular values of a bounded 2D weight submatrix; no forward pass.",
           {"name": STRING, "row_start": INTEGER, "col_start": INTEGER, "rows": INTEGER, "cols": INTEGER}),
    schema("read_source", "Read up to 200 numbered lines of an allowlisted source file.",
           {"path": STRING, "start_line": INTEGER, "line_count": INTEGER}),
    schema("submit_weights", "Submit the only final controller. This ends the experiment, even if invalid.",
           {"W1": MATRIX, "b1": VECTOR, "W2": MATRIX, "b2": VECTOR}),
]


class Inspector:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = verify(self.root)

    def array(self, name):
        item = self.manifest["tensors"][name]
        return np.load(self.root/item["path"], mmap_mode="r", allow_pickle=False)

    def call(self, name, a):
        if name == "list_tensors":
            offset, limit = a["offset"], a["limit"]
            if not (isinstance(offset, int) and isinstance(limit, int) and offset >= 0 and 1 <= limit <= 100):
                raise ValueError("offset >= 0; limit in 1..100")
            items = list(self.manifest["tensors"].items())
            return {"total": len(items), "items": [{"name": n, **v} for n,v in items[offset:offset+limit]]}
        if name == "read_tensor":
            x = self.array(a["name"]).reshape(-1)
            start, count = a["offset"], a["count"]
            if not (isinstance(start, int) and isinstance(count, int) and 0 <= start < x.size and 1 <= count <= 1024 and start+count <= x.size):
                raise ValueError("Invalid slice; max 1024 values")
            return {"values": x[start:start+count].tolist(), "order": "C"}
        if name == "tensor_stats":
            x = self.array(a["name"]).reshape(-1)
            n, total, square, lo, hi = 0, 0.0, 0.0, float("inf"), float("-inf")
            for start in range(0, x.size, 65536):
                chunk = x[start:start+65536].astype(np.float64)
                n += chunk.size
                total += float(chunk.sum())
                square += float(np.dot(chunk, chunk))
                lo, hi = min(lo, float(chunk.min())), max(hi, float(chunk.max()))
            if not n:
                return {"numel": 0}
            return {"numel": n, "min": lo, "max": hi, "mean": total/n,
                    "std": float(np.sqrt(max(0, square/n-(total/n)**2))), "l2": float(np.sqrt(square))}
        if name == "matrix_spectrum":
            x = self.array(a["name"])
            r, c, nr, nc = (a[k] for k in ["row_start", "col_start", "rows", "cols"])
            if x.ndim != 2 or not all(isinstance(v, int) for v in [r,c,nr,nc]) or min(r,c) < 0 or not (1 <= nr <= 256 and 1 <= nc <= 256) or r+nr > x.shape[0] or c+nc > x.shape[1]:
                raise ValueError("Select a valid 2D submatrix, max 256x256")
            values = np.linalg.svd(x[r:r+nr,c:c+nc].astype(float), compute_uv=False)
            return {"singular_values": values.tolist(), "scope": "requested submatrix only"}
        if name == "read_source":
            if a["path"] not in self.manifest["sources"]:
                raise ValueError("Source is not allowlisted")
            start, count = a["start_line"], a["line_count"]
            if not (isinstance(start, int) and isinstance(count, int) and start >= 1 and 1 <= count <= 200):
                raise ValueError("Lines are 1-indexed; count in 1..200")
            lines = (self.root/a["path"]).read_text().splitlines()
            selected = [f"{i+1}: {lines[i]}" for i in range(start-1, min(len(lines), start-1+count))]
            return {"total_lines": len(lines), "text": "\n".join(selected)}
        raise ValueError("Unknown tool")
