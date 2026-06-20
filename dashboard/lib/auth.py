"""Firebase email/password authentication for the dashboard.

Authentication uses the Firebase Identity Toolkit REST API directly (no pyrebase
dependency). The public entry point is require_auth(): app.py calls it once, in
place, and it runs the full gate — logout handling, cookie-restore, ?guest=true
auto-login, the dev bypass, and finally the login page + st.stop() until the user
is signed in. All session-state and query-param keys are preserved verbatim.
"""

import hashlib
import hmac
import json
from pathlib import Path

import streamlit as st

# assets/ sits next to the dashboard package (dashboard/assets), one level up from
# this module (dashboard/lib/auth.py).
_LOGIN_CSS = Path(__file__).resolve().parent.parent / "assets" / "login.css"


def _secrets_section(name: str) -> dict:
    """Return a Streamlit secrets section, or an empty dict when secrets are absent."""
    try:
        section = st.secrets.get(name, {})
    except st.errors.StreamlitSecretNotFoundError:
        return {}
    return dict(section) if section else {}


def _make_auth_token(email: str) -> str:
    firebase_cfg = _secrets_section("firebase")
    secret = firebase_cfg.get("apiKey", "").encode()
    if not secret:
        return ""
    return hmac.new(secret, email.lower().encode(), hashlib.sha256).hexdigest()


def _set_auth_cookie(email: str):
    token = _make_auth_token(email)
    st.html(
        f"""<script>
        document.cookie = "auth_email={email}; path=/; max-age=86400; SameSite=Strict";
        document.cookie = "auth_token={token}; path=/; max-age=86400; SameSite=Strict";
        </script>"""
    )


def _clear_auth_cookie():
    st.html(
        """<script>
        document.cookie = "auth_email=; path=/; max-age=0";
        document.cookie = "auth_token=; path=/; max-age=0";
        </script>"""
    )


def _firebase_sign_in_with_password(email: str, password: str) -> dict:
    """Authenticate with Firebase Email/Password using the Identity Toolkit REST API."""
    firebase_cfg = _secrets_section("firebase")
    api_key = firebase_cfg.get("apiKey", "")
    if not api_key:
        st.error(
            "**Firebase config missing.** Add a `[firebase]` section to "
            "`.streamlit/secrets.toml`. See the dashboard README for setup instructions."
        )
        st.stop()

    import urllib.error
    import urllib.request

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = json.dumps(
        {"email": email, "password": password, "returnSecureToken": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw).get("error", {}).get("message", raw)
        except json.JSONDecodeError:
            detail = raw
        raise RuntimeError(detail) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"NETWORK_ERROR: {exc.reason}") from exc


def _render_login():
    """Render a full-page login form. Authenticates via Firebase Email/Password."""
    st.markdown(_LOGIN_CSS.read_text(encoding="utf-8"), unsafe_allow_html=True)

    st.markdown(
        "<div class='login-logo-row'>"
        "<svg width='48' height='48' viewBox='0 0 100 100' fill='none' xmlns='http://www.w3.org/2000/svg' style='flex-shrink:0;'>"
        "<path d='M25 85 L25 40 L15 15 L30 30 L50 25 L70 30 L85 15 L75 40 L75 85 Z' "
        "fill='#e84545' opacity='0.9'/>"
        "<circle cx='38' cy='50' r='5' fill='#000'/>"
        "<circle cx='62' cy='50' r='5' fill='#000'/>"
        "<path d='M45 60 Q50 65 55 60' stroke='#000' stroke-width='2' fill='none'/>"
        "<line x1='20' y1='55' x2='38' y2='52' stroke='#000' stroke-width='1.5'/>"
        "<line x1='20' y1='60' x2='38' y2='58' stroke='#000' stroke-width='1.5'/>"
        "<line x1='62' y1='52' x2='80' y2='55' stroke='#000' stroke-width='1.5'/>"
        "<line x1='62' y1='58' x2='80' y2='60' stroke='#000' stroke-width='1.5'/>"
        "</svg>"
        "<div>"
        "<span class='login-logo-dsk'>InsiderGuard AI</span>"
        "<span class='login-logo-sub'>Made By: Data Structure Kittens</span>"
        "</div>"
        "</div>"
        "<hr class='login-divider'>"
        "<p class='login-heading'>Analyst Portal &mdash; Sign In</p>",
        unsafe_allow_html=True,
    )

    email = st.text_input(
        "Email address",
        placeholder="analyst@organisation.com",
        key="login_email",
    )
    password = st.text_input(
        "Password",
        type="password",
        placeholder="••••••••",
        key="login_password",
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.session_state.get("_login_error"):
        st.error(st.session_state["_login_error"])

    _col_sign_in, _col_demo = st.columns([3, 2])
    with _col_sign_in:
        _clicked_signin = st.button("SIGN IN", use_container_width=True, type="primary")
    with _col_demo:
        _clicked_demo = st.button("DEMO ACCESS", use_container_width=True)

    if _clicked_demo:
        st.query_params["guest"] = "true"
        st.rerun()

    if _clicked_signin:
        if not email or not password:
            st.session_state["_login_error"] = "Email and password are required."
            st.rerun()
        else:
            try:
                user = _firebase_sign_in_with_password(email, password)
                st.session_state["authenticated"] = True
                st.session_state["auth_user_email"] = user.get("email", email)
                st.session_state["auth_id_token"] = user.get("idToken", "")
                st.session_state["_set_cookie"] = True
                st.session_state.pop("_login_error", None)
                st.rerun()
            except Exception as exc:
                msg = str(exc)
                if any(k in msg for k in ("INVALID_PASSWORD", "EMAIL_NOT_FOUND", "INVALID_LOGIN_CREDENTIALS", "INVALID_EMAIL")):
                    st.session_state["_login_error"] = "Invalid email or password."
                elif "TOO_MANY_ATTEMPTS_TRY_LATER" in msg:
                    st.session_state["_login_error"] = "Too many failed attempts. Try again later."
                elif "USER_DISABLED" in msg:
                    st.session_state["_login_error"] = "This account has been disabled."
                elif "OPERATION_NOT_ALLOWED" in msg:
                    st.session_state["_login_error"] = "Email/password sign-in is not enabled in Firebase."
                elif "EMAIL_NOT_VERIFIED" in msg:
                    st.session_state["_login_error"] = "Please verify your email address before signing in."
                else:
                    st.session_state["_login_error"] = f"Sign-in failed: {msg}"
                st.rerun()

    st.markdown(
        "<p class='login-footer'>UEBA Insider Threat Detection &mdash; Senior Design Project &middot; 2026</p>",
        unsafe_allow_html=True,
    )


def require_auth() -> None:
    """Run the full auth gate in place. Stops the script (login page) until signed in.

    Order preserved from the original inline gate: logout → cookie-restore → guest
    → dev bypass → login + st.stop() → set-cookie pop on first login.
    """
    # ── Handle logout via query param (set by the fixed sidebar button) ──
    _is_logout = st.query_params.get("logout") == "true"
    if _is_logout:
        st.session_state.clear()
        st.query_params.clear()
        _clear_auth_cookie()

    # ── Restore session from auth cookie (survives page refresh) ──
    if not _is_logout and not st.session_state.get("authenticated", False):
        _cookies = st.context.cookies
        _c_email = _cookies.get("auth_email", "")
        _c_token = _cookies.get("auth_token", "")
        _expected_token = _make_auth_token(_c_email) if _c_email else ""
        if _c_email and _c_token and _expected_token and hmac.compare_digest(_c_token, _expected_token):
            st.session_state["authenticated"] = True
            st.session_state["auth_user_email"] = _c_email

    # ── Guest auto-login via ?guest=true URL parameter ──
    # LinkedIn / demo link: https://dsk-insider-threat-detection.streamlit.app/?guest=true
    # Requires [guest] section in .streamlit/secrets.toml with email + password.
    if not _is_logout and not st.session_state.get("authenticated", False):
        if st.query_params.get("guest", "").lower() == "true":
            _guest_cfg = _secrets_section("guest")
            _g_email    = _guest_cfg.get("email", "")
            _g_password = _guest_cfg.get("password", "")
            if _g_email and _g_password:
                try:
                    _g_user = _firebase_sign_in_with_password(_g_email, _g_password)
                    st.session_state["authenticated"] = True
                    st.session_state["auth_user_email"] = "Guest"
                    st.session_state["auth_id_token"]   = _g_user.get("idToken", "")
                    st.session_state["is_guest"]         = True
                    # Don't persist guest session in cookie — ?guest=true re-auths on refresh
                except Exception:
                    pass  # fall through to login page

    # ── Auth gate — show login and stop until the user has signed in ──
    _dev_bypass = _secrets_section("dev").get("bypass_auth", False)
    if _dev_bypass and not st.session_state.get("authenticated", False):
        st.session_state["authenticated"] = True
        st.session_state["auth_user_email"] = "dev@local"
    if not st.session_state.get("authenticated", False):
        _render_login()
        st.stop()

    # ── Set auth cookie after first successful login ──
    if st.session_state.pop("_set_cookie", False):
        _set_auth_cookie(st.session_state.get("auth_user_email", ""))
