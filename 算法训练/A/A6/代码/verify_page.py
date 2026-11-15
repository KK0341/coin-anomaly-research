"""Scoped A6 asset and offline-browser checks. Uses installed Playwright + Edge, no server."""
from pathlib import Path
import json, hashlib
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'结果/review_manifest.json').read_text(encoding='utf8'))
assets=json.loads((ROOT/'结果/图像来源.json').read_text(encoding='utf8'))
results={'dataset_id':data['dataset_id'],'target_count':len(data['samples'])}
assert len(data['samples'])==30 and len({s['certificate'] for s in data['samples']})==30
targets={s['certificate'] for s in data['samples']}
for s in data['samples']:
    assert s['split']=='val' and len(s['images'])==2
    assert 1<=len(s['references'])<=3
    for r in s['references']:
        assert r['split']=='train' and r['truth']['label']==0 and r['certificate'] not in targets
        assert len(r['images'])==2
for rel,meta in assets.items():
    p=ROOT/rel
    with Image.open(p) as im: assert im.size==(1024,1024); im.verify()
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
results['original_resolution_images']=len(assets)
results['copied_image_hashes_match']=True
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
    context=browser.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
    page=context.new_page(); errors=[]
    page.on('pageerror',lambda err:errors.append(str(err)))
    page.goto((ROOT/'人工观察.html').as_uri())
    page.wait_for_function('document.querySelectorAll(".coin").length===4 && [...document.querySelectorAll(".coin")].every(x=>x.complete && x.naturalWidth===1024)')
    assert not page.locator('#truth').is_visible()
    assert not page.locator('#showTruth').is_checked() and not page.locator('#showHeat').is_checked()
    assert page.locator('#sampleList button').count()==30
    blind=page.locator('body').inner_text()
    assert '选样组：' not in blind and '目标真实标签：' not in blind
    results['default_labels_and_heatmaps_hidden']=True
    # Walk all 30 paired targets and both paired references through file://.
    for i in range(30):
        page.locator(f'.sample-button[data-index="{i}"]').click()
        for j in range(len(data['samples'][i]['references'])):
            page.locator(f'#refButtons button[data-ref="{j}"]').click()
            page.wait_for_function('[...document.querySelectorAll(".coin")].every(x=>x.complete && x.naturalWidth===1024)')
    results['all_target_and_reference_pairs_load_offline']=True
    page.locator('.sample-button[data-index="0"]').click()
    page.locator('.panel').first.hover();page.mouse.wheel(0,-500)
    page.wait_for_timeout(150)
    views=page.evaluate('[...document.querySelectorAll(".panel")].map(p=>p.view)')
    assert views[0]['z']>1 and views[0]==views[1]
    p=page.locator('.panel').first.bounding_box();x=p['x']+p['width']/2;y=p['y']+p['height']/2
    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+45,y+20);page.mouse.up()
    moved=page.evaluate('[...document.querySelectorAll(".panel")].map(p=>p.view)')
    assert moved[0]['x']!=views[0]['x'] and moved[0]==moved[1]
    page.locator('#linked').uncheck();page.locator('.panel').first.hover();page.mouse.wheel(0,-200);page.wait_for_timeout(100)
    views=page.evaluate('[...document.querySelectorAll(".panel")].map(p=>p.view)')
    assert views[0]['z']!=views[1]['z']
    results['zoom_pan_link_and_unlink']=True
    page.locator('#resetZoom').click();page.locator('#linked').check()
    page.locator('#nativeZoom').click()
    native=page.evaluate('document.querySelector(".panel").view.z*document.querySelector(".panel").clientWidth')
    assert abs(native-1024)<1
    page.locator('#resetZoom').click();page.locator('.focus').first.click()
    assert page.locator('.focused').count()==1
    page.keyboard.press('Escape');assert page.locator('.focused').count()==0
    page.locator('#showTruth').check();assert '选样组：' in page.locator('#truth').inner_text()
    page.locator('#showHeat').check();assert page.locator('.heat').first.is_visible()
    page.locator('#showTruth').uncheck();page.locator('#showHeat').uncheck()
    page.locator('input[name="status"]').nth(3).check();page.locator('#part').fill('正面底板 / 测试');page.locator('#notes').fill('QA 临时记录：逗号, 引号" 与换行\n中文');page.locator('#localizable').select_option('无法判断')
    with page.expect_download() as dl: page.locator('#exportJson').click()
    exported=json.loads(Path(dl.value.path()).read_text(encoding='utf8'))
    first=exported['reviews'][0]
    assert first['status']=='当前照片无法判断' and first['label_seen'] and first['heatmap_seen']
    assert len(exported['reviews'])==30 and first['split']=='val'
    with page.expect_download() as dl: page.locator('#exportCsv').click()
    csvdata=Path(dl.value.path()).read_text(encoding='utf-8-sig')
    assert '""' in csvdata and 'QA 临时记录' in csvdata
    page.reload();page.wait_for_selector('#notes')
    assert page.locator('#notes').input_value().startswith('QA 临时记录')
    assert not page.locator('#showTruth').is_checked()
    page.evaluate('localStorage.clear()');page.reload()
    page.locator('#importFile').set_input_files({'name':'observations.json','mimeType':'application/json','buffer':json.dumps(exported,ensure_ascii=False).encode()})
    page.wait_for_function('document.querySelector("#notes").value.startsWith("QA 临时记录")')
    results['annotation_persistence_csv_json_and_import']=True
    page.evaluate('localStorage.clear()');page.reload()
    page.wait_for_function('[...document.querySelectorAll(".coin")].every(x=>x.complete && x.naturalWidth===1024)')
    page.screenshot(path=str(ROOT/'结果/网页预览.png'),full_page=True)
    page.set_viewport_size({'width':800,'height':1000});page.screenshot(path=str(ROOT/'结果/窄屏预览.png'),full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    results['javascript_errors']=errors;assert not errors
    browser.close()
(ROOT/'结果/网页核验.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(results,ensure_ascii=False,indent=2))
