"""Session lifecycle, isolation and resource management tests."""

import unittest

from PIL import Image

from backend.session_manager import (
    SessionManager,
    SessionNotFoundError,
    SessionCapacityError,
)


class FakeClock:

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeAgent:

    def __init__(self):
        self.release_count = 0
        self.reset_count = 0

    def release_models(self):
        self.release_count += 1

    def reset(self):
        self.reset_count += 1


class TestSessionManager(unittest.TestCase):

    def setUp(self):

        self.clock = FakeClock()

        self.manager = SessionManager(
            max_sessions=2,
            ttl_seconds=60,
            agent_factory=FakeAgent,
            clock=self.clock,
        )

        self.addCleanup(self.manager.close)

    def test_unique_session_identifiers(self):

        first = self.manager.create()
        second = self.manager.create()

        self.assertNotEqual(first, second)
        self.assertGreaterEqual(len(first), 32)

        self.assertEqual(
            self.manager.get_statistics()["active_sessions"],
            2,
        )

    def test_session_capacity(self):

        first = self.manager.create()
        self.manager.create()

        with self.assertRaises(SessionCapacityError):
            self.manager.create()

        self.manager.delete(first)

        replacement = self.manager.create()

        self.assertNotEqual(first, replacement)

    def test_session_expiration(self):

        session_id = self.manager.create()

        with self.manager.lease(session_id) as agent:
            pass

        self.clock.advance(60)

        self.assertEqual(
            self.manager.cleanup_expired(),
            1,
        )

        self.assertEqual(agent.reset_count, 1)

        with self.assertRaises(SessionNotFoundError):
            with self.manager.lease(session_id):
                pass

    def test_activity_refreshes_expiration(self):

        session_id = self.manager.create()

        self.clock.advance(45)

        with self.manager.lease(session_id):
            self.clock.advance(30)

        self.clock.advance(59)

        self.assertEqual(
            self.manager.cleanup_expired(),
            0,
        )

        self.clock.advance(1)

        self.assertEqual(
            self.manager.cleanup_expired(),
            1,
        )

    def test_model_release_on_session_switch(self):

        first = self.manager.create()
        second = self.manager.create()

        with self.manager.lease(first) as agent_a:
            pass

        with self.manager.lease(first):
            pass

        self.assertEqual(agent_a.release_count, 0)

        with self.manager.lease(second) as agent_b:
            pass

        self.assertEqual(agent_a.release_count, 1)
        self.assertEqual(agent_b.release_count, 0)
        self.assertEqual(agent_a.reset_count, 0)

        with self.manager.lease(first):
            pass

        self.assertEqual(agent_b.release_count, 1)

    def test_delete_releases_session(self):

        session_id = self.manager.create()

        with self.manager.lease(session_id) as agent:
            pass

        self.manager.delete(session_id)

        self.assertEqual(agent.reset_count, 1)

        with self.assertRaises(SessionNotFoundError):
            with self.manager.lease(session_id):
                pass

    def test_operation_failure_preserves_session(self):

        session_id = self.manager.create()

        with self.assertRaises(ValueError):

            with self.manager.lease(session_id):
                raise ValueError("Simulated operation failure")

        with self.manager.lease(session_id) as agent:
            self.assertEqual(agent.reset_count, 0)

    def test_shutdown(self):

        first = self.manager.create()
        second = self.manager.create()

        with self.manager.lease(first) as agent_a:
            pass

        with self.manager.lease(second) as agent_b:
            pass

        self.manager.close()
        self.manager.close()

        self.assertEqual(agent_a.reset_count, 1)
        self.assertEqual(agent_b.reset_count, 1)

        with self.assertRaises(RuntimeError):
            self.manager.create()

    def test_real_agent_session_isolation(self):

        manager = SessionManager()

        self.addCleanup(manager.close)

        first = manager.create()
        second = manager.create()

        image = Image.new("RGB", (320, 240), "white")

        with manager.lease(first) as agent:

            agent.upload_image(image)
            agent.choose_objective("speed")
            agent.choose_models()

            proposal = agent.choose_threshold(0.5)

            token = proposal["confirmation_token"]

        with manager.lease(second) as agent:

            self.assertFalse(
                agent.get_context()["image_uploaded"]
            )

            self.assertFalse(
                agent.get_context()["analysis_completed"]
            )

            with self.assertRaises(
                (ValueError, RuntimeError)
            ):
                agent.confirm(token, approved=True)

        with manager.lease(first) as agent:

            self.assertTrue(
                agent.get_context()["image_uploaded"]
            )

            self.assertFalse(
                agent.get_context()["analysis_completed"]
            )

            agent.confirm(token, approved=False)

    def test_invalid_configuration(self):

        for value in (0, -1, True, 1.5):

            with self.assertRaises(ValueError):
                SessionManager(max_sessions=value)

        for value in (0, -1, float("nan"), float("inf")):

            with self.assertRaises(ValueError):
                SessionManager(ttl_seconds=value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
