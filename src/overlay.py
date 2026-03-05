"""Floating overlay panel for showing transcription status."""

from AppKit import (
    NSPanel, NSColor, NSFont, NSTextField, NSMakeRect,
    NSWindowStyleMaskBorderless, NSBackingStoreBuffered,
    NSFloatingWindowLevel, NSScreen,
)
from PyObjCTools.AppHelper import callAfter


class TranscriptionOverlay:
    """A borderless floating panel that shows partial transcription text."""

    def __init__(self):
        screen = NSScreen.mainScreen().frame()
        width, height = 600, 80
        x = (screen.size.width - width) / 2
        y = screen.size.height * 0.12

        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, width, height),
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setIgnoresMouseEvents_(True)
        panel.setHasShadow_(True)

        content = panel.contentView()
        content.setWantsLayer_(True)
        layer = content.layer()
        layer.setCornerRadius_(12)
        layer.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.1, 0.1, 0.75).CGColor()
        )

        label = NSTextField.alloc().initWithFrame_(NSMakeRect(16, 10, width - 32, height - 20))
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setTextColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.5))
        label.setFont_(NSFont.systemFontOfSize_(16))
        label.setStringValue_("")
        content.addSubview_(label)

        self._panel = panel
        self._label = label

    def show(self, text=""):
        def _do():
            self._label.setStringValue_(text)
            self._panel.orderFront_(None)
        callAfter(_do)

    def update_text(self, text):
        def _do():
            self._label.setStringValue_(text)
        callAfter(_do)

    def hide(self):
        def _do():
            self._panel.orderOut_(None)
        callAfter(_do)
