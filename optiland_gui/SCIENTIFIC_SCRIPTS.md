# Scientific scripts and GUI extensions

The script editor has two execution actions:

- **Run Script / F5** uses the existing interactive Console. Its `connector` and
  `iface` are the real GUI objects, and existing GUI extensions retain synchronous
  access. Long code in this console can block the window.
- **Scientific Run** uses a separate process and an independent copy of the
  current optic. The window and optical preview queue remain available. Each run
  starts with a fresh local namespace; **Restart** also resets the process and
  its imported module state. Editor text survives Stop and Restart.

Scientific code receives `optic`, `np` (NumPy), `be` (Optiland backend), `optiland`
and `gui`. Ordinary Python imports work. This is execution of trusted local code,
not a security sandbox. GUI widgets, `connector`, `iface` and arbitrary Console
variables are not copied into the scientific process.

```python
optic.surfaces[1].comment = "Scientific candidate"
optic.trace(0, 0, 0.55, 101, "line_y")
print(optic.surfaces.y)
gui.show_panel("viewer")
```

The result and output appear under **Scientific Results**. A successful run does
not change the current design. **Apply Script Result** explicitly installs the
candidate and records Undo. If the document has changed, including a comment,
the result cannot replace those edits; run the script again against the current
document. A backend change, Stop or Restart also revokes application permission.

The `gui` recorder supports `show_panel("viewer" | "analysis" | "lens_editor")`
and `refresh_views()` (repaint retained views). At most 32 commands can be recorded.
They are applied only after successful, current execution and cannot call arbitrary
Qt code or return widget objects.

Source text is limited to 1 MiB. Standard output and errors each retain at most
64 KiB, 1,000 lines and 4,096 characters per line, with a truncation marker. Updates
are rate limited. Stop cooperatively cancels where possible and terminates an
unresponsive scientific process after the calculation cancellation grace period.
The next run starts a replacement process when needed. All scientific processes
participate in asynchronous application shutdown.
