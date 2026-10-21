"""Stop before training when Drive holds duplicate same-name folders on a result path.

Two Colab runtimes (e.g. DINO and CLIP) that create the SAME new folder at nearly
the same time each get their own copy; the mounted path then silently resolves to
either one, and resume/skip logic sees the wrong run (2026-10-21 pool v2 incident).
The mount cannot show duplicates, so this reads folder names via the Drive API.
Read-only metadata scope; nothing on Drive is created, renamed or deleted.
"""
from pathlib import Path

FOLDER = 'application/vnd.google-apps.folder'
DRIVE_ROOT = Path('/content/drive/MyDrive')


class DuplicateFolderError(RuntimeError):
    pass


def drive_service():
    import google.auth
    from googleapiclient.discovery import build
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive.metadata.readonly'])
    return build('drive', 'v3', credentials=credentials, cache_discovery=False)


def child_folders(service, parent, name):
    quoted = name.replace('\\', '\\\\').replace("'", "\\'")
    query = f"'{parent}' in parents and name = '{quoted}' and mimeType = '{FOLDER}' and trashed = false"
    found, token = [], None
    while True:
        page = service.files().list(q=query, fields='nextPageToken,files(id,name,createdTime)',
                                    pageToken=token, pageSize=100).execute()
        found += page.get('files', [])
        token = page.get('nextPageToken')
        if not token:
            return found


def check(paths, model_tag, service=None, drive_root=DRIVE_ROOT):
    """Raise on any duplicate level. Return the first missing level of each path and
    whether that level is SHARED (its name lacks model_tag), i.e. racy if two
    runtimes start together."""
    service = service or drive_service()
    missing = []
    for path in paths:
        parts = Path(path).relative_to(drive_root).parts
        parent = 'root'
        for depth, name in enumerate(parts):
            found = child_folders(service, parent, name)
            prefix = '/'.join(parts[:depth+1])
            if len(found) > 1:
                rows = '\n'.join(f"  id={f['id']} created={f.get('createdTime')}" for f in sorted(found, key=lambda f: f.get('createdTime', '')))
                raise DuplicateFolderError(f'Duplicate Drive folders named MyDrive/{prefix}:\n{rows}\n'
                    'Rename the copy you do not want (Drive UI) so only one remains, then rerun this cell.')
            if not found:
                missing.append(dict(path=str(path), first_missing=prefix, shared=model_tag not in name))
                break
            parent = found[0]['id']
    for row in missing:
        if row['shared']:
            print(f"WARNING: MyDrive/{row['first_missing']} does not exist yet and is shared by all models. "
                  'Start the other model only after this notebook has created it.', flush=True)
    print('Drive folder check passed: no duplicate result folders.', flush=True)
    return missing
