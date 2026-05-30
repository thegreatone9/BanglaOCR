#!/usr/bin/env python3
"""
Playwright test script for the Bangla OCR Streamlit app at http://localhost:8501
"""

import time
import os
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

SCREENSHOT_DIR = "/Users/musakhan/.gemini/antigravity/brain/de2d1790-50cc-4f14-89b7-cbb2c51448ea"
APP_URL = "http://localhost:8501"

def run_test():
    with sync_playwright() as p:
        print("Launching Chromium browser...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        # Step 1: Navigate to the app
        print(f"\n[Step 1] Navigating to {APP_URL}...")
        page.goto(APP_URL, wait_until="networkidle", timeout=30000)
        print("  ✓ Navigation successful")

        # Step 2: Wait for Streamlit app to fully load
        print("\n[Step 2] Waiting for app to load...")
        try:
            # Wait for the Streamlit app container
            page.wait_for_selector(".stApp", timeout=15000)
            print("  ✓ Streamlit app container found")
        except PlaywrightTimeout:
            print("  ✗ .stApp not found, trying alternative...")

        # Extra wait for React rendering
        time.sleep(3)

        # Step 3: Take initial screenshot
        screenshot_path = f"{SCREENSHOT_DIR}/screenshot_initial.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"\n[Step 3] Initial screenshot saved: {screenshot_path}")

        # Step 4: Get page title and content
        title = page.title()
        print(f"\n[Step 4] Page title: {title}")

        # Step 5: Look for key elements and text
        print("\n[Step 5] Inspecting page content...")
        try:
            content = page.content()
            if "Bangla" in content or "bangla" in content.lower():
                print("  ✓ 'Bangla' text found in page")
            if "OCR" in content or "ocr" in content.lower():
                print("  ✓ 'OCR' text found in page")
            if "error" in content.lower() or "Error" in content:
                print("  ⚠ 'Error' text found in page - possible model loading issue")
            if "model" in content.lower():
                print("  ✓ 'model' text found in page")
        except Exception as e:
            print(f"  ✗ Content check failed: {e}")

        # Step 6: Get all visible text
        print("\n[Step 6] Visible text on page:")
        try:
            text = page.evaluate("() => document.body.innerText")
            # Print first 2000 chars
            print(text[:2000] if text else "(empty)")
        except Exception as e:
            print(f"  Error getting text: {e}")

        # Step 7: Find canvas element
        print("\n[Step 7] Looking for canvas element...")
        canvas = page.query_selector("canvas")
        if canvas:
            bbox = canvas.bounding_box()
            print(f"  ✓ Canvas found at: {bbox}")

            # Draw on the canvas using mouse events
            print("\n[Step 8] Drawing on canvas...")
            try:
                cx = bbox["x"]
                cy = bbox["y"]

                # Draw a diagonal stroke (like part of ক)
                page.mouse.move(cx + 50, cy + 30)
                page.mouse.down()
                for i in range(20):
                    page.mouse.move(cx + 50 + i*3, cy + 30 + i*4)
                    time.sleep(0.02)
                page.mouse.up()

                time.sleep(0.3)

                # Draw a vertical stroke
                page.mouse.move(cx + 100, cy + 20)
                page.mouse.down()
                for i in range(20):
                    page.mouse.move(cx + 100, cy + 20 + i*4)
                    time.sleep(0.02)
                page.mouse.up()

                time.sleep(0.3)

                # Draw a horizontal stroke
                page.mouse.move(cx + 60, cy + 60)
                page.mouse.down()
                for i in range(15):
                    page.mouse.move(cx + 60 + i*4, cy + 60)
                    time.sleep(0.02)
                page.mouse.up()

                print("  ✓ Drawing strokes completed")
                time.sleep(0.5)

                # Screenshot after drawing
                draw_screenshot = f"{SCREENSHOT_DIR}/screenshot_after_draw.png"
                page.screenshot(path=draw_screenshot, full_page=True)
                print(f"  ✓ Post-draw screenshot saved: {draw_screenshot}")

            except Exception as e:
                print(f"  ✗ Drawing failed: {e}")
        else:
            print("  ✗ No canvas element found")

        # Step 9: Find and click Predict button
        print("\n[Step 9] Looking for Predict button...")
        try:
            # Try various selectors for the Predict button
            predict_btn = None
            selectors = [
                "button:has-text('Predict')",
                "button:has-text('predict')",
                "[data-testid='baseButton-secondary']:has-text('Predict')",
                ".stButton button",
            ]
            for sel in selectors:
                try:
                    btn = page.query_selector(sel)
                    if btn:
                        predict_btn = btn
                        print(f"  ✓ Found Predict button with selector: {sel}")
                        break
                except:
                    pass

            if predict_btn:
                predict_btn.click()
                print("  ✓ Clicked Predict button")
                time.sleep(4)  # Wait for prediction

                # Final screenshot
                results_screenshot = f"{SCREENSHOT_DIR}/screenshot_results.png"
                page.screenshot(path=results_screenshot, full_page=True)
                print(f"  ✓ Results screenshot saved: {results_screenshot}")

                # Check for results
                text_after = page.evaluate("() => document.body.innerText")
                print("\n[Step 10] Page text after prediction:")
                print(text_after[:2000] if text_after else "(empty)")

                if "Predicted" in text_after or "predicted" in text_after:
                    print("\n  ✓ PREDICTION RESULT FOUND!")
                elif "ক" in text_after or "খ" in text_after or "গ" in text_after:
                    print("\n  ✓ Bangla character in results!")
                else:
                    print("\n  ⚠ No explicit prediction result found in text")
            else:
                print("  ✗ Predict button not found")
                # List all buttons
                all_buttons = page.query_selector_all("button")
                print(f"  Found {len(all_buttons)} buttons total:")
                for btn in all_buttons[:10]:
                    try:
                        print(f"    - '{btn.inner_text()}'")
                    except:
                        pass

        except Exception as e:
            print(f"  ✗ Predict step failed: {e}")

        # Final summary screenshot
        final_screenshot = f"{SCREENSHOT_DIR}/screenshot_final.png"
        page.screenshot(path=final_screenshot, full_page=True)
        print(f"\n[Final] Screenshot saved: {final_screenshot}")

        browser.close()
        print("\n✓ Test completed! Browser closed.")
        print(f"\nScreenshot files saved to: {SCREENSHOT_DIR}")

if __name__ == "__main__":
    run_test()
