import unittest

from ink_service import create_session, hash_password, valid_session, verify_password


class LocalAuthBehavior(unittest.TestCase):
    def test_given_a_password_when_hashed_then_plaintext_is_not_stored(self):
        stored = hash_password("correct horse battery staple")
        self.assertNotIn("correct horse battery staple", stored)
        self.assertTrue(verify_password("correct horse battery staple", stored))
        self.assertFalse(verify_password("wrong password", stored))

    def test_given_a_session_when_signature_is_changed_then_it_is_rejected(self):
        session = create_session("artist@example.com")
        self.assertEqual(valid_session(session), "artist@example.com")
        encoded, signature = session.split(".", 1)
        self.assertIsNone(valid_session(f"{encoded}.{signature[:-1]}0"))


if __name__ == "__main__":
    unittest.main()