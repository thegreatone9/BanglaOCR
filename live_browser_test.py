#!/usr/bin/env python3
"""
Live browser test - connects to the already-running Chrome with remote debugging.
Uses PointerEvent (not MouseEvent) to work with Fabric.js inside streamlit-drawable-canvas.
"""

import time
import sys
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

SCREENSHOT_DIR = "/Users/musakhan/.gemini/antigravity/brain/de2d1790-50cc-4f14-89b7-cbb2c51448ea"
APP_URL = "http://localhost:8501"
CDP_URL = "http://localhost:9222"

def fire_pointer_events(page, canvas_x, canvas_y, strokes):
    """
    Fire PointerEvents on the canvas using evaluate_script.
    Fabric.js uses pointer events, not mouse events.
    strokes: list of lists of (x, y) points relative to canvas top-left
    """
    script = f"""
    async function drawStrokes() {{
        const canvas = document.querySelector('canvas');
        if (!canvas) return 'ERROR: No canvas found';
        
        const rect = canvas.getBoundingClientRect();
        console.log('Canvas rect:', JSON.stringify(rect));
        
        function sleep(ms) {{ return new Promise(r => setTimeout(r, ms)); }}
        
        function firePointer(type, x, y, pressure=0) {{
            const evt = new PointerEvent(type, {{
                bubbles: true,
                cancelable: true,
                view: window,
                clientX: rect.left + x,
                clientY: rect.top + y,
                pointerId: 1,
                pointerType: 'mouse',
                isPrimary: true,
                pressure: pressure,
                buttons: type === 'pointerup' ? 0 : 1,
                button: type === 'pointerup' ? 0 : 0
            }});
            canvas.dispatchEvent(evt);
        }}
        
        const strokes = {strokes};
        
        for (const stroke of strokes) {{
            if (stroke.length === 0) continue;
            const [sx, sy] = stroke[0];
            firePointer('pointermove', sx, sy, 0);
            await sleep(20);
            firePointer('pointerdown', sx, sy, 0.5);
            await sleep(30);
            
            for (let i = 1; i < stroke.length; i++) {{
                const [x, y] = stroke[i];
                firePointer('pointermove', x, y, 0.5);
                await sleep(15);
            }}
            
            const last = stroke[stroke.length - 1];
            firePointer('pointerup', last[0], last[1], 0);
            await sleep(100);
        }}
        
        return 'Drawing complete! Canvas: ' + rect.width + 'x' + rect.height + 
               ' at (' + Math.round(rect.left) + ',' + Math.round(rect.top) + ')';
    }}
    return drawStrokes();
    """
    return page.evaluate(script)

def run_live_test():
    with sync_playwright() as p:
        print(f"Connecting to live Chrome at {CDP_URL}...")
        try:
            browser = p.chromium.connect_over_cdp(CDP_URL)
            print("✓ Connected to live Chrome browser!")
            
            # Get or create the page with the app
            contexts = browser.contexts
            page = None
            for ctx in contexts:
                for pg in ctx.pages:
                    print(f"  Found page: {pg.url}")
                    if "localhost:8501" in pg.url or pg.url == "about:blank":
                        page = pg
            
            if not page:
                print("  Creating new page...")
                ctx = browser.new_context()
                page = ctx.new_page()
            
            # Navigate to app
            print(f"\nNavigating to {APP_URL}...")
            page.bring_to_front()
            if "localhost:8501" not in page.url:
                page.goto(APP_URL, wait_until="networkidle", timeout=30000)
            else:
                print(f"  Already on the app: {page.url}")
            
            # Wait for Streamlit to fully load
            print("Waiting for app to fully load...")
            page.wait_for_selector(".stApp", timeout=20000)
            time.sleep(3)
            print(f"  ✓ App loaded! Title: {page.title()}")
            
            # Take initial screenshot
            ss1 = f"{SCREENSHOT_DIR}/live_screenshot_initial.png"
            page.screenshot(path=ss1)
            print(f"  ✓ Initial screenshot: {ss1}")
            
            # Find canvas dimensions
            canvas_info = page.evaluate("""
                () => {
                    const canvas = document.querySelector('canvas');
                    if (!canvas) return null;
                    const rect = canvas.getBoundingClientRect();
                    return {
                        width: canvas.width,
                        height: canvas.height,
                        rectWidth: rect.width,
                        rectHeight: rect.height,
                        left: rect.left,
                        top: rect.top
                    };
                }
            """)
            
            if not canvas_info:
                print("  ✗ No canvas found! Printing page structure...")
                print(page.evaluate("() => document.body.innerHTML[:2000]"))
                return
            
            print(f"\n✓ Canvas found: {canvas_info}")
            w = canvas_info['rectWidth']
            h = canvas_info['rectHeight']
            
            # Define strokes for a simple Bangla-like character (ক approximation)
            # Vertical stroke on the right
            # Horizontal stroke through the middle
            # Curved stroke on the left
            strokes = []
            
            # Stroke 1: Vertical line (right side of ক)
            stroke1 = [[int(w*0.6), int(h*0.15 + i*(h*0.65/20))] for i in range(21)]
            strokes.append(stroke1)
            
            # Stroke 2: Horizontal line (matra/headline)
            stroke2 = [[int(w*0.2 + i*(w*0.55/20)), int(h*0.3)] for i in range(21)]
            strokes.append(stroke2)
            
            # Stroke 3: Left curved part
            import math
            stroke3 = []
            for i in range(20):
                t = i / 19.0
                x = int(w*0.35 - w*0.1 * math.sin(t * math.pi))
                y = int(h*0.3 + t * h*0.45)
                stroke3.append([x, y])
            strokes.append(stroke3)
            
            print(f"\nDrawing on canvas with PointerEvents ({len(strokes)} strokes)...")
            result = fire_pointer_events(page, 0, 0, strokes)
            print(f"  Result: {result}")
            time.sleep(1)
            
            # Take post-draw screenshot
            ss2 = f"{SCREENSHOT_DIR}/live_screenshot_drawn.png"
            page.screenshot(path=ss2)
            print(f"  ✓ Post-draw screenshot: {ss2}")
            
            # Click Predict button
            print("\nLooking for Predict button...")
            predict_btn = page.query_selector("button:has-text('Predict')")
            if predict_btn:
                print("  ✓ Found Predict button — clicking!")
                predict_btn.click()
                print("  ✓ Clicked! Waiting for results...")
                time.sleep(5)
                
                # Take results screenshot
                ss3 = f"{SCREENSHOT_DIR}/live_screenshot_results.png"
                page.screenshot(path=ss3)
                print(f"  ✓ Results screenshot: {ss3}")
                
                # Read results text
                page_text = page.evaluate("() => document.body.innerText")
                print("\n=== Page text after prediction ===")
                print(page_text[:3000])
                
                if "⚠️" in page_text and "empty" in page_text:
                    print("\n⚠️  Canvas still reads as empty — PointerEvents not captured by Fabric.js")
                    print("   Trying alternative: directly manipulate Fabric.js canvas object...")
                    
                    # Try via Fabric.js internal API
                    fabric_result = page.evaluate("""
                    async () => {
                        // Try to find Fabric canvas instance
                        const canvasEl = document.querySelector('canvas');
                        if (!canvasEl) return 'No canvas';
                        
                        // Fabric.js attaches to upper canvas
                        const upperCanvas = document.querySelector('.upper-canvas') || canvasEl;
                        const rect = upperCanvas.getBoundingClientRect();
                        
                        function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
                        
                        // Try mouse events as fallback
                        function fireAll(x, y, type) {
                            [MouseEvent, PointerEvent].forEach(EventClass => {
                                upperCanvas.dispatchEvent(new EventClass(type, {
                                    bubbles: true, cancelable: true, view: window,
                                    clientX: rect.left + x, clientY: rect.top + y,
                                    buttons: 1, button: 0, pressure: 0.5,
                                    pointerId: 1, pointerType: 'mouse', isPrimary: true
                                }));
                            });
                        }
                        
                        fireAll(100, 50, 'pointerdown');
                        fireAll(100, 50, 'mousedown');
                        await sleep(50);
                        for (let i = 0; i < 30; i++) {
                            fireAll(100 + i*2, 50 + i*2, 'pointermove');
                            fireAll(100 + i*2, 50 + i*2, 'mousemove');
                            await sleep(20);
                        }
                        fireAll(160, 110, 'pointerup');
                        fireAll(160, 110, 'mouseup');
                        
                        return 'Fired events on upper-canvas at ' + rect.left + ',' + rect.top;
                    }
                    """)
                    print(f"   Fabric attempt: {fabric_result}")
                    time.sleep(1)
                    
                    predict_btn = page.query_selector("button:has-text('Predict')")
                    if predict_btn:
                        predict_btn.click()
                        time.sleep(4)
                        ss4 = f"{SCREENSHOT_DIR}/live_screenshot_results2.png"
                        page.screenshot(path=ss4)
                        print(f"  ✓ Second results screenshot: {ss4}")
                        page_text2 = page.evaluate("() => document.body.innerText")
                        if "Predicted" in page_text2 or any(c in page_text2 for c in "কখগঘঙচছজঝঞ"):
                            print("\n✅ PREDICTION WORKED!")
                        else:
                            print("\nPage results:")
                            # Find the results section
                            for line in page_text2.split('\n'):
                                if any(kw in line for kw in ['Predict', 'Result', 'character', 'word', '⚠', '✅', 'ক', 'খ']):
                                    print(f"  {line}")
                else:
                    print("\n✅ PREDICTION RESULT RECEIVED!")
            else:
                print("  ✗ Predict button not found")
            
            print("\n✓ Live browser test complete! Browser remains open for you to see.")
            print("\nScreenshots saved:")
            print(f"  - {ss1}")
            print(f"  - {ss2}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_live_test()
