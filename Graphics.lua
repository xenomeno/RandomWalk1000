dofile("Bitmap.lua")

local function clamp(x, min, max)
  if x < min then
    return min
  elseif x > max then
    return max
  else
    return x
  end
end

function DrawGraphs(bmp, funcs_data, descr)
  descr = descr or {}
  
  local div = descr.div or 10
  local skip_KP = descr.skip_KP
  local write_frames, write_name = descr.write_frames, descr.write_name
  local frames_step = descr.frames_step or 1
  local start_x, start_y = descr.start_x or 0, descr.start_y or 0
  local width, height = descr.width or bmp.width, descr.height or bmp.height
  local axis_x_format, axis_y_format = descr.axis_x_format, descr.axis_y_format
  local int_x, int_y = descr.int_x, descr.int_y
  
  local order = {}
  if descr.sort_cmp then
    local entries = {}
    for name, func in pairs(funcs_data.funcs) do
      table.insert(entries, {name = name, sort_idx = func.sort_idx})
    end
    table.sort(entries, descr.sort_cmp)
    for k, entry in ipairs(entries) do
      order[k] = entry.name
    end
  else
    for name in pairs(funcs_data.funcs) do
      table.insert(order, name)
    end
    table.sort(order)
  end
  
  local min_x, min_y, max_x, max_y
  local any_pt = funcs_data.funcs[next(funcs_data.funcs)][1]
  if descr.center_x then
    min_x, max_x = descr.center_x, descr.center_x
  else
    min_x, max_x = any_pt.x, any_pt.x
  end
  if descr.center_y then
    min_y, max_y = descr.center_y, descr.center_y
  else
    min_y, max_y = any_pt.y, any_pt.y
  end
  for _, name in ipairs(order) do
    local func_points = funcs_data.funcs[name]
    for _, pt in ipairs(func_points) do
      local x, y = pt.x, pt.y
      min_x = (x < min_x) and x or min_x
      min_y = (y < min_y) and y or min_y
      max_x = (x > max_x) and x or max_x
      max_y = (y > max_y) and y or max_y
    end
  end
  
  local size_x = int_x and math.ceil(max_x - min_x) or (max_x - min_x)
  local size_y = int_y and math.ceil(max_y - min_y) or (max_y - min_y)
  if descr.scale_uniformly then
    if size_x > size_y then
      size_y = size_x
    else
      size_x = size_y
    end
  end
  
  local center_x, center_y = descr.center_x or min_x, descr.center_y or min_y
  local spacing_x, spacing_y = width // (div + 2), height // (div + 2)
  local scale_x, scale_y = div * spacing_x / size_x, div * spacing_y / size_y
  local Ox = start_x + spacing_x
  local Oy = start_y + height - spacing_y
  local axes_color = funcs_data.axes_color or RGB_GRAY
  local bars_y_padding = descr.bars_y_padding or 5

  -- draw coordinate system
  if not axis_x_format then
    axis_x_format = int_x and "%d" or "%.2f"
  end
  if not axis_y_format then
    axis_y_format = int_y and "%d" or "%.2f"
  end
  bmp:DrawLine(Ox - spacing_x // 2, Oy, Ox + div * spacing_x + spacing_x // 2, Oy, axes_color)
  local axis_Y_col = descr.right_axis_Y and (Ox + div * spacing_x + spacing_x // 4) or Ox
  bmp:DrawLine(axis_Y_col, Oy + spacing_y // 2, axis_Y_col, Oy - div * spacing_y - spacing_y // 2, axes_color)
  local metric_x, metric_y = spacing_x // div, spacing_y // div
  for k = 1, div do
    bmp:DrawLine(Ox + k * spacing_x, Oy - metric_y, Ox + k * spacing_x, Oy + metric_y, axes_color)
    bmp:DrawLine(axis_Y_col - metric_x, Oy - k * spacing_y, axis_Y_col + metric_x, Oy - k * spacing_y, axes_color)
    local text = int_x and string.format(axis_x_format, k * size_x // div + center_x) or string.format(axis_x_format, k * size_x / div + center_x)
    local tw, th = bmp:MeasureText(text)
    bmp:DrawText(Ox + k * spacing_x - tw // 2, Oy + 2 * metric_y, text, axes_color)
    text = int_y and string.format(axis_y_format, k * size_y // div + center_y) or string.format(axis_y_format, k * size_y / div + center_y)
    tw, th = bmp:MeasureText(text)
    bmp:DrawText(descr.right_axis_Y and (width - tw - 5) or start_x, Oy - k * spacing_y - th // 2, text, axes_color)
  end
  local level_y_text = int_y and string.format(axis_y_format, center_y) or string.format(axis_y_format, center_y)
  local tw, th = bmp:MeasureText(level_y_text)
  bmp:DrawText(descr.right_axis_Y and (width - tw - 5) or start_x, Oy - th - 2, level_y_text, axes_color)
  
  -- draw graphs
  local box_size = 2
  local name_x = spacing_x + 10
  for _, name in ipairs(order) do
    local func_points = funcs_data.funcs[name]
    local color = func_points.color
    local last_x, last_y
    local frame = 0
    for idx, pt in ipairs(func_points) do
      local x = math.floor(Ox + scale_x * (pt.x - center_x))
      local y = math.floor(Oy - scale_y * (pt.y - center_y))
      if descr.bars then
        y = Min(y, Oy - bars_y_padding)
      end
      if last_x and last_y then
        bmp:DrawLine(last_x, last_y, x, y, color)
      end
      if not skip_KP then
        bmp:DrawBox(x - box_size, y - box_size, x + box_size, y + box_size, color)
      end
      if pt.text then
        local w, h = bmp:MeasureText(pt.text)
        bmp:DrawText(x - w // 2, y - h - 2, pt.text, color)
      end
      if descr.bars and last_x and x > last_x then
        for bar_x = last_x, x do
          local bar_y = last_y + (y - last_y) * (bar_x - last_x) // (x - last_x)
          bmp:DrawLine(bar_x, Min(bar_y, Oy - bars_y_padding), bar_x, Oy - bars_y_padding, color)
        end
      end
      last_x, last_y = x, y
      if write_frames and (not write_name or name == write_name) and (idx % frames_step == 0 or idx == #func_points) then
        frame = frame + 1
        local filename = string.format("%s_%s%04d.bmp", write_frames, not write_name and (name and "_") or "", frame)
        print(string.format("Writing '%s' ...", filename))
        bmp:WriteBMP(filename)
      end
    end
    if #func_points == 1 then
      bmp:SetPixel(last_x, last_y, color)
      if not skip_KP then
        bmp:DrawBox(last_x - box_size, last_y - box_size, last_x + box_size, last_y + box_size, color)
      end
      if write_frames then
        local filename = string.format("%s_%s1.bmp", write_frames, not write_name and (name .. "_") or "")
        print(string.format("Writing '%s' ...", filename))
        bmp:WriteBMP(filename)
      end
    end
    local w, h = bmp:MeasureText(name)
    if descr.right_axis_Y then
      bmp:DrawText(width - name_x - w - 5, start_y + height - h, name, color)
    else
      bmp:DrawText(start_x + name_x, start_y + height - h, name, color)
    end
    name_x = name_x + w + 30
  end
  
  if funcs_data.name_y then
    if descr.right_axis_Y then
      local w, h = bmp:MeasureText(funcs_data.name_y)
      bmp:DrawText(width - w - start_x, start_y + 5, funcs_data.name_y, axes_color)
    else
      bmp:DrawText(start_x + 5, start_y + 5, funcs_data.name_y, axes_color)
    end
  end
  if funcs_data.name_x then
    local w, h = bmp:MeasureText(funcs_data.name_x)
    bmp:DrawText(start_x + width - w - 5, start_y + height - h * 2 - 5, funcs_data.name_x, axes_color)
  end
  
  return function(pt)
    return {x = Ox + math.floor(scale_x * (pt.x - center_x)), y = Oy - math.floor(scale_y * (pt.y - center_y))}
  end
end
