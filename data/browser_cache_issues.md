# Resolving Web App Loading and Display Issues

If parts of the user dashboard or login buttons do not load, it is often due to outdated static files cached in the browser.

## Recommended Fixes
1. **Force Refresh (Hard Reload)**:
   - Windows: Press `Ctrl + F5` or `Ctrl + Shift + R`.
   - Mac: Press `Cmd + Shift + R`.
2. **Clear Application Cache**:
   - Open browser Developer Tools (`F12`).
   - Go to **Application** tab (Chrome) or **Storage** tab (Firefox).
   - Click **Clear Site Data** or **Clear Storage**.
3. **Use Incognito/Private Mode**:
   - Test if the login/loading issue persists in an incognito window. If it works there, clearing cookies and cache will resolve the primary browser issue.
