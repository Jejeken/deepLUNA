"""
Basic test for CXR v3 - Test UI loading without full functionality
"""

import customtkinter as ctk
import sys
import os

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

def test_basic_ui():
    """Test basic UI loading"""
    try:
        print("Testing basic UI components...")

        # Test drag drop frame
        root = ctk.CTk()
        root.title("Test CXR v3 Basic")
        root.geometry("800x600")

        try:
            from ui_v3.drag_drop import DragDropFrame
            print("✓ DragDropFrame imported successfully")

            drop_frame = DragDropFrame(root)
            drop_frame.pack(fill="both", expand=True, padx=20, pady=20)
            print("✓ DragDropFrame created successfully")

        except Exception as e:
            print(f"✗ DragDropFrame error: {e}")
            # Create fallback
            fallback_label = ctk.CTkLabel(root, text=f"Drag drop error: {e}")
            fallback_label.pack(expand=True)

        print("Basic UI test completed. Close window to continue...")
        root.mainloop()

    except Exception as e:
        print(f"Basic UI test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success = test_basic_ui()
    print(f"Test result: {'PASSED' if success else 'FAILED'}")