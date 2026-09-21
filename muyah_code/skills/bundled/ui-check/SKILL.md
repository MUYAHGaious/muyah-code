---
name: ui-check
description: Use after building or changing a web UI, or when asked to check a page works - opens it in a real browser, walks the changed flow and reports what a user would see.
---
# Check a web UI in a real browser

Tests passing is not the same as the page working. Look at it the way a user would.

1. **Start the app** if it is not running (the project's dev command, e.g. `npm run dev`), with Bash in the
   background (`run_in_background`), and wait until it prints its URL. Note the URL.
2. **Open the browser tools:** call `Browser` once if the browser_* tools are not there yet.
3. **Load the page:** `browser_navigate` to the URL, then `browser_snapshot` to read it. The snapshot is
   the page as text, with refs you pass to `browser_click` / `browser_type`.
4. **Walk the changed flow** step by step: click, type, submit. After each step take a new snapshot and check
   that what changed is what should have changed.
5. **Look for errors:** `browser_console_messages` (JavaScript errors) and `browser_network_requests`
   (failed calls, 4xx/5xx).
6. **Screenshot** (`browser_take_screenshot`) only if you can see images and the layout matters.
7. **Clean up:** `browser_close`, and stop the dev server you started.
8. **Report:** each step, what you saw, and a verdict per changed behaviour (works / broken / could not
   check). For anything broken, show the evidence (snapshot text, console error) before proposing a fix.
