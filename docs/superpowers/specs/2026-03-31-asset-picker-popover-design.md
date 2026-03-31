# Asset Picker Popover Design

**Date:** 2026-03-31
**Scope:** `react/src/components/chat/ChatTextarea.tsx`

## Summary

Replace the existing `DropdownMenu`-based material asset picker with a `Popover`-based floating panel that includes a name search input and a thumbnail grid view.

## UI Structure

- **Trigger:** Existing `Library` icon button (unchanged)
- **Panel size:** `w-80` (320px), `max-h-96`, inner content scrollable
- **Top (fixed):** Search `Input` with search icon, placeholder "搜索素材…"
- **Content grid:** `grid-cols-3`, each cell:
  - Image: thumbnail (`object-cover`, rounded)
  - Video: `Video` icon + name
  - Audio: `Music` icon + name
  - Name displayed below thumbnail, truncated to 1 line
- **Empty state:** "无匹配素材" centered text when no results
- **On item click:** call existing `addAssetToChat(rec)` then close Popover

## State Changes

| State | Type | Purpose |
|---|---|---|
| `assetPickerOpen` | `boolean` | Controls Popover open state |
| `assetSearchQuery` | `string` | Search input value, reset on close |

## Filtering Logic

```
activeFiles = materialFiles.filter(r => r.disk_name && r.serve_url && r.status === 'Active')
filteredFiles = activeFiles.filter(r => r.name.toLowerCase().includes(assetSearchQuery.toLowerCase()))
```

## Import Changes

Add to existing imports:
- `Popover, PopoverContent, PopoverTrigger` from `@/components/ui/popover`
- `Input` from `@/components/ui/input`
- `Search` icon from `lucide-react`

Remove:
- `DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger` (if not used elsewhere in the file)

## Implementation Notes

- Data source unchanged: existing `useQuery(['material-files'])`
- No new API calls needed
- All filtering is client-side
- `assetSearchQuery` should reset to `''` when Popover closes (`onOpenChange`)
- Show the picker button even when `materialFiles.length === 0` (Popover will show empty state)
