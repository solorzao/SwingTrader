# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/swing_trader/ui/app.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'swing_trader',
        'swing_trader.ui',
        'swing_trader.ui.views',
        'swing_trader.ui.views.signals',
        'swing_trader.ui.views.training',
        'swing_trader.ui.views.backtest',
        'swing_trader.ui.views.models',
        'swing_trader.data',
        'swing_trader.features',
        'swing_trader.models',
        'swing_trader.signals',
        'swing_trader.backtest',
        'swing_trader.training',
        'sklearn',
        'sklearn.ensemble',
        'sklearn.model_selection',
        'xgboost',
        'torch',
        'pandas',
        'numpy',
        'yfinance',
        'mlflow',
        'optuna',
        'joblib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SwingTrader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)
