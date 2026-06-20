"""Unit tests for the deterministic parts of dashboard/lib/auth.py.

Covers the cookie-token HMAC and the secrets accessor — pure logic that does not
need the Streamlit runtime. The interactive login flow and the require_auth() gate
are exercised by the AppTest smoke test (dev-bypass path), not here.
"""

import hashlib
import hmac
from unittest import mock

from dashboard.lib import auth


class TestMakeAuthToken:
    def test_empty_secret_returns_empty_string(self):
        with mock.patch.object(auth, "_secrets_section", return_value={}):
            assert auth._make_auth_token("analyst@example.com") == ""

    def test_token_is_hmac_sha256_of_lowercased_email(self):
        with mock.patch.object(auth, "_secrets_section", return_value={"apiKey": "topsecret"}):
            token = auth._make_auth_token("Analyst@Example.com")
        expected = hmac.new(
            b"topsecret", b"analyst@example.com", hashlib.sha256
        ).hexdigest()
        assert token == expected

    def test_token_is_case_insensitive_on_email(self):
        with mock.patch.object(auth, "_secrets_section", return_value={"apiKey": "k"}):
            assert auth._make_auth_token("A@B.COM") == auth._make_auth_token("a@b.com")

    def test_token_matches_via_compare_digest(self):
        # The gate validates cookies with hmac.compare_digest against this token.
        with mock.patch.object(auth, "_secrets_section", return_value={"apiKey": "k"}):
            t1 = auth._make_auth_token("u@x.com")
            t2 = auth._make_auth_token("u@x.com")
        assert hmac.compare_digest(t1, t2)


class TestSecretsSection:
    def test_missing_section_returns_empty_dict(self):
        # st.secrets behaves like a mapping; .get returns {} for an absent section.
        fake_secrets = mock.MagicMock()
        fake_secrets.get.return_value = {}
        with mock.patch.object(auth.st, "secrets", fake_secrets):
            assert auth._secrets_section("nope") == {}

    def test_present_section_returned_as_dict(self):
        fake_secrets = mock.MagicMock()
        fake_secrets.get.return_value = {"email": "g@x.com", "password": "pw"}
        with mock.patch.object(auth.st, "secrets", fake_secrets):
            out = auth._secrets_section("guest")
        assert out == {"email": "g@x.com", "password": "pw"}
        assert isinstance(out, dict)
