import unittest
from unittest.mock import patch

from fastagent.recording.video import VideoRecorder


class FakeRecordingClient:
    instances = []

    def __init__(self, base_url=None):
        self.end_result = b"" if not self.instances else b"synthetic-video"
        self.closed = False
        self.instances.append(self)

    async def start_recording(self):
        return True

    async def end_recording(self, dest=None):
        return self.end_result

    async def close(self):
        self.closed = True


class VideoRecorderFailureStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_recording_result_does_not_leave_recorder_stuck(self):
        FakeRecordingClient.instances = []
        with patch("fastagent.recording.video.RecordingClient", FakeRecordingClient):
            recorder = VideoRecorder("synthetic.mp4", base_url="http://synthetic.invalid")

            self.assertTrue(await recorder.start())
            self.assertFalse(await recorder.stop())
            self.assertFalse(recorder.is_recording)
            self.assertIsNone(recorder._client)

            self.assertTrue(await recorder.start())
            self.assertTrue(await recorder.stop())
            self.assertFalse(recorder.is_recording)
            self.assertEqual(len(FakeRecordingClient.instances), 2)
            self.assertTrue(all(client.closed for client in FakeRecordingClient.instances))


if __name__ == "__main__":
    unittest.main()
