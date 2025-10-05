from collections import defaultdict, deque
import numpy as np

class PersonTracker:
    def __init__(self, max_history=5):
        self.persons = {}
        self.action_history = defaultdict(lambda: deque(maxlen=max_history))
        self.next_id = 0

    def match_and_update(self, boxes, keypoints_list, actions):
        results = {}
        for i, box in enumerate(boxes):
            matched_id = None
            max_iou = 0
            for pid, prev_box in self.persons.items():
                iou = self.bbox_iou(box, prev_box)
                if iou > 0.3 and iou > max_iou:
                    matched_id = pid
                    max_iou = iou
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
            self.persons[matched_id] = box
            # Append the action for the current frame to this person's history
            self.action_history[matched_id].append(actions[i])
            smoothed_action = max(set(self.action_history[matched_id]),
                                  key=self.action_history[matched_id].count)
            results[matched_id] = {
                "bbox": box,
                "keypoints": keypoints_list[i].tolist(),
                "action": smoothed_action
            }
        return results

    @staticmethod
    def bbox_iou(b1, b2):
        x1, y1, x2, y2 = b1
        x1b, y1b, x2b, y2b = b2
        xi1, yi1 = max(x1, x1b), max(y1, y1b)
        xi2, yi2 = min(x2, x2b), min(y2, y2b)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2b - x1b) * (y2b - y1b)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
