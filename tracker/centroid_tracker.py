from collections import OrderedDict
import numpy as np


class Track:
    def __init__(self, object_id, centroid, bbox):
        self.object_id = int(object_id)
        self.centroid = tuple(map(int, centroid))
        self.bbox = tuple(map(int, bbox))
        self.disappeared = 0
        self.previous_y = int(centroid[1])
        self.counted = False
        self.stable_class = "unknown"
        self.class_votes = {"soybean": 0, "black_bean": 0, "unknown": 0}


class CentroidTracker:
    def __init__(self, max_disappeared=12, max_distance=55):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.max_disappeared = int(max_disappeared)
        self.max_distance = float(max_distance)

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = Track(self.next_object_id, centroid, bbox)
        self.next_object_id += 1

    def deregister(self, object_id):
        self.objects.pop(object_id, None)

    def update(self, detections):
        # detections: list of {"centroid": (x,y), "bbox": (x,y,w,h), "class": str}
        if len(detections) == 0:
            for oid in list(self.objects):
                self.objects[oid].disappeared += 1
                if self.objects[oid].disappeared > self.max_disappeared:
                    self.deregister(oid)
            return list(self.objects.values())

        if len(self.objects) == 0:
            for d in detections:
                self.register(d["centroid"], d["bbox"])
            self._apply_classes(detections)
            return list(self.objects.values())

        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[i].centroid for i in object_ids])
        input_centroids = np.array([d["centroid"] for d in detections])
        distances = np.linalg.norm(object_centroids[:, None, :] - input_centroids[None, :, :], axis=2)

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            track = self.objects[oid]
            track.previous_y = track.centroid[1]
            track.centroid = tuple(map(int, detections[col]["centroid"]))
            track.bbox = tuple(map(int, detections[col]["bbox"]))
            track.disappeared = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distances.shape[0])) - used_rows
        unused_cols = set(range(distances.shape[1])) - used_cols

        for row in unused_rows:
            oid = object_ids[row]
            self.objects[oid].disappeared += 1
            if self.objects[oid].disappeared > self.max_disappeared:
                self.deregister(oid)
        for col in unused_cols:
            self.register(detections[col]["centroid"], detections[col]["bbox"])

        self._apply_classes(detections)
        return list(self.objects.values())

    def _apply_classes(self, detections):
        # Assign each detection to nearest currently tracked object.
        for d in detections:
            if not self.objects:
                break
            c = np.array(d["centroid"])
            track = min(self.objects.values(),
                        key=lambda t: np.linalg.norm(np.array(t.centroid) - c))
            label = d.get("class", "unknown")
            track.class_votes[label] = track.class_votes.get(label, 0) + 1
            track.stable_class = max(track.class_votes, key=track.class_votes.get)
