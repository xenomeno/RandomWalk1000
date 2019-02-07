dofile("Bitmap.lua")
dofile("Graphics.lua")
dofile("CommonAI.lua")

local STATES                  = 1000
local STATE_LEFT              = 0
local STATE_RIGHT             = STATES + 1
local ACTIONS                 = 100
local ACTION_TERMINATE        = 0
local REWARD_LEFT             = -1.0
local REWARD_RIGHT            = 1.0
local EPSILON                 = 0.01
local GROUPS                  = 10
local GROUP_SIZE              = STATES // GROUPS
local EPISODES                = 100000
local ALPHA                   = 0.00002

-- TD(n) properties for graph
local GROUPS_TDn              = 20
local GROUP_SIZE_TDn          = 50
local ACTIONS_TDn             = 50
local RUNS                    = 100
local ALPHA_STEP              = 0.1
local MAX_N                   = 512
local FIRST_EPISODES          = 10
local CLAMP_MIN_RMS           = 0.40
local CLAMP_MAX_RMS           = 1.00

-- Polynomial VS Fourier
local BASIS_ORDERS            = {5, 10, 20}
local ALPHA_POLYNOMIAL        = 0.0001
local ALPHA_FOURIER           = 0.00005
local BASIS_RUNS              = 30
local BASIS_EPISODES          = 5000
local BASIS_CLAMP_MIN_RMS     = 0.0
local BASIS_CLAMP_MAX_RMS     = 1.0

-- Tile Coding
local ALPHA_TILINGS           = 0.0001
local TILINGS                 = 50
local TILE_WIDTH              = 200
local TILE_OFFSET             = 4
local TILING_RUNS             = 30
local TILING_EPISODES         = 10000
local TILING_CLAMP_MIN_RMS    = 0.0
local TILING_CLAMP_MAX_RMS    = 1.0

local IMAGE_WIDTH             = 1000
local IMAGE_HEIGHT            = 1000
local IMAGE_FILENAME_MC       = "RandomWalk1000/RandomWalk1000_MC.bmp"
local IMAGE_FILENAME_TD0      = "RandomWalk1000/RandomWalk1000_TD(0).bmp"
local IMAGE_FILENAME_TDn      = "RandomWalk1000/RandomWalk1000_TDn.bmp"
local IMAGE_FILENAME_BASIS    = "RandomWalk1000/RandomWalk1000_Basis.bmp"
local IMAGE_FILENAME_TILING   = "RandomWalk1000/RandomWalk1000_Tiling.bmp"

local function CreateProb(actions)
  actions = actions or ACTIONS
  
  local prob = {}
  local action_prob = 1.0 / (2.0 * actions)
  for s = 1, STATES do
    prob[s] = {}
    for a = 1, actions do
      if s - a >= 1 then
        prob[s][-a] = action_prob
      end
      if s + a <= STATES then
        prob[s][a] = action_prob
      end
    end
    local miss = (s < actions) and (actions - s + 1) or ((s > STATES - actions) and (actions  -(STATES - s)) or 0)
    prob[s][ACTION_TERMINATE] = miss * action_prob
  end
  
  return prob
end

local function GetAction(actions)
  return math.random(1, (actions or ACTIONS)) * ((math.random() < 0.5) and 1 or -1)
end

local function GetNextState(s, a)
  local next_s, reward = s + a, 0
  if next_s <= STATE_LEFT then
    next_s = STATE_LEFT
    reward = REWARD_LEFT
    a = ACTION_TERMINATE
  elseif next_s >= STATE_RIGHT then
    next_s = STATE_RIGHT
    reward = REWARD_RIGHT
    a = ACTION_TERMINATE
  end
  
  return next_s, reward, a
end

local function ComputeTrueValue(prob, actions)
  actions = actions or ACTIONS
  
  ResetTime()
  
  local true_V = {}
  -- initial guesses
  for s = STATE_LEFT, STATE_RIGHT do
    true_V[s] = REWARD_LEFT + (REWARD_RIGHT - REWARD_LEFT) * (s - STATE_LEFT) / (STATE_RIGHT - STATE_LEFT)
  end
  local prob = CreateProb()
  
  local delta, iterations = 1.0, 0
  while Abs(delta) > EPSILON do
    iterations = iterations + 1
    if iterations % 100 == 0 then
      print(string.format("True Value Calculation: Iteration #%d", iterations))
    end
    local old_V = table.copy(true_V)
    for s = 1, STATES do
      local new_value = 0.0
      local actions_left, actions_right = -Min(s - 1, actions), Min(STATES - s, actions)
      for a = actions_left, actions_right do
        local next_s = (a == ACTION_TERMINATE) and ((s < ACTIONS) and STATE_LEFT or STATE_RIGHT) or GetNextState(s, a)
        new_value = new_value + prob[s][a] * true_V[next_s]
      end
      true_V[s] = new_value
    end
    delta = 0.0
    for s = 1, STATES do
      delta = delta + Abs(true_V[s] - old_V[s])
    end
  end
  true_V[STATE_LEFT], true_V[STATE_RIGHT] = 0.0, 0.0
  print(string.format("True Value for %d states: %d iterations, Time: %.2f", STATES, iterations, GetTime()))
  
  return true_V
end

local function NormalizeDistribution(distribution)
  local total = 0
  for s = 1, STATES do
    total = total + (distribution[s] or 0)
  end
  for s = 1, STATES do
    distribution[s] = (distribution[s] or 0) / total
  end  
end

local function RMS(V1, V2)
  local rms = 0.0
  for s = 1, STATES do
    rms = rms + Sqr(V1[s] - V2[s])
  end
  
  return math.sqrt(rms / STATES)
end

local function NormalizeRMS(rms, denom)
  local scale = 1.0 / denom
  for index, value in ipairs(rms) do
    rms[index] = value * scale
  end
end

local ValueFunction =
{
  size = false,
  weights = false,
}

function ValueFunction:new(o)
  o = o or {}
  setmetatable(o, self)
  self.__index = self
  
  if o.size then
    o:Init()
  end
  
  return o
end

function ValueFunction:Init()
  local weights = {}
  for d = 0, self.size + 1 do
    weights[d] = 0.0
  end
  self.weights = weights
end

local StateAggregationValueFunction = ValueFunction:new{}

function StateAggregationValueFunction:GetValue(s)
  local d
  if s == STATE_LEFT then
    d = 1
  elseif s == STATE_RIGHT then
    d = self.size + 1
  else
    d = ((s - 1) // self.group_size) + 1
  end
  
  return self.weights[d], 1, d
end

function StateAggregationValueFunction:Update(s, delta, d)
  if not d then
    local _, _, dim = self:GetValue(s)
    d = dim
  end
  self.weights[d] = self.weights[d] + delta
end

local BasisValueFunction = ValueFunction:new{}

function BasisValueFunction:Init()
  local basis, weights = {}, {}
  for d = 1, self.size do
    if self.basis_type == "polynomials" then
      basis[d] = function(s) return math.pow(s, d - 1) end
    else
      basis[d] = function(s) return math.cos(d * math.pi * s) end
    end
    weights[d] = 0.0
  end
  self.basis, self.weights = basis, weights
end

function BasisValueFunction:GetValue(s)
  local norm_s = s / STATES
  local basis, weights = self.basis, self.weights
  local weight = 0.0
  for dim, func in ipairs(basis) do
    weight = weight + weights[dim] * func(norm_s)
  end
  
  return weight, 1
end

function BasisValueFunction:Update(s, delta)
  local norm_s = s / STATES
  local funcs, weights = self.basis, self.weights
  for dim, func in ipairs(funcs) do
    weights[dim] = weights[dim] + delta * func(norm_s)
  end
end

local TileCoding = ValueFunction:new{}

function TileCoding:Init()
  self.tile_offset = self.tile_offset or 0
  
  local space = STATES + self.size * self.tile_offset
  local tiling_size = space // self.tile_width + ((space % self.tile_width == 0) and 0 or 1)
  local weights = {}
  for tiling = 1, self.size do
    weights[tiling] = {}
    for tile = 1, tiling_size do
      weights[tiling][tile] = 0.0
    end
  end
  self.weights = weights
end

function TileCoding:GetValue(s)
  local tile_width, tile_offset = self.tile_width, self.tile_offset
  local weights = self.weights
  local value = 0.0
  local start_s = -self.size * tile_offset + 1
  for tiling, tiling_weights in ipairs(weights) do
    local dim = 1 + (s - start_s) // tile_width
    value = value + tiling_weights[dim]
    start_s = start_s + tile_offset
  end
  
  return value, 1
end

function TileCoding:Update(s, delta)
  local tile_width, tile_offset = self.tile_width, self.tile_offset
  local weights = self.weights
  local start_s = -self.size * tile_offset + 1
  for tiling, tiling_weights in ipairs(weights) do
    local dim = 1 + (s - start_s) // tile_width
    tiling_weights[dim] = tiling_weights[dim] + delta
    start_s = start_s + tile_offset
  end
end

local function GradientMonteCarlo(alpha, value_function, max_episodes, rms, true_V)
  max_episodes = max_episodes or EPISODES
  
  local distribution = {}
  for episode = 1, max_episodes do
    local s = STATES // 2
    local ep, G = {}
    while s ~= STATE_LEFT and s ~= STATE_RIGHT do
      table.insert(ep, s)
      local a = GetAction()
      local next_s, reward = GetNextState(s, a)
      s, G = next_s, reward
    end
    distribution[s] = (distribution[s] or 0) + 1
    for _, s in ipairs(ep) do
      local value, dw, dim = value_function:GetValue(s)
      local delta = alpha * (G - value) * dw
      value_function:Update(s, delta, dim)
      distribution[s] = (distribution[s] or 0) + 1
    end
    if rms then
      local V = {}
      for s = 1, STATES do
        V[s] = value_function:GetValue(s)
      end
      rms[episode] = (rms[episode] or 0.0) + RMS(V, true_V)
    end
  end
  NormalizeDistribution(distribution)
  
  return distribution
end

local function SemiGradientTD0(alpha, value_function)
  local distribution = {}
  for episode = 1, EPISODES do
    local s = STATES // 2
    while s ~= STATE_LEFT and s ~= STATE_RIGHT do
      local a = GetAction()
      local next_s, reward = GetNextState(s, a)
      local value, dw, dim = value_function:GetValue(s)
      local next_value = value_function:GetValue(next_s)
      local delta = alpha * (reward + next_value - value) * dw
      value_function:Update(s, delta, dim)
      distribution[s] = (distribution[s] or 0) + 1
      s = next_s
    end
  end
  NormalizeDistribution(distribution)
  
  return distribution
end

local function SemiGradientTDn(alpha, n, gamma, value_function, max_episodes, true_V)
  max_episodes = max_episodes or EPISODES
  
  local rms = 0.0
  local store_state, store_reward, store_index = {}, {}, 1
  for episode = 1, max_episodes do
    local s = STATES // 2
    local t, terminated = 1, false
    store_state[1], store_reward[1], store_index = s, 0, 1
    repeat
      if not terminated then
        local a = GetAction(ACTIONS_TDn)
        local next_s, reward = GetNextState(s, a)
        table.insert(store_state, next_s)
        table.insert(store_reward, reward)
        terminated = (next_s == STATE_LEFT or next_s == STATE_RIGHT) and (t + 1)
      end
      local t_update = t - n + 1
      if t_update >= 1 then
        local t_end = terminated and Min(t_update + n, terminated) or (t_update + n)
        local G, gamma_mul = 0.0, 1.0
        for i = t_update + 1, t_end do
          G = G + gamma_mul * store_reward[i]
          gamma_mul = gamma_mul * gamma
        end
        if not terminated or t_update + n <= terminated then
          local value = value_function:GetValue(store_state[t_update + n])
          G = G + gamma_mul * value
        end
        local state_to_update = store_state[t_update]
        local value, dw, dim = value_function:GetValue(state_to_update)
        local delta = alpha * (G - value) * dw
        value_function:Update(state_to_update, delta, dim)
      end
      t = t + 1
      s = store_state[#store_state]
    until terminated and t_update == terminated - 1
    local V = {}
    for s = 1, STATES do
      V[s] = value_function:GetValue(s)
    end
    rms = rms + RMS(V, true_V)
  end
  
  return rms
end

local function CreateMCTD0Graphs(true_V)
  local graphs = {funcs = {}, name_x = "State", name_y = "Value"}
  local graphs_right = {funcs = {}, name_x = "State", name_y = "Distribution Scale"}
  
  ResetTime()
  math.randomseed(0)
  local gmc_V = StateAggregationValueFunction:new{size = GROUPS, group_size = GROUP_SIZE}
  local distribution_mc = GradientMonteCarlo(ALPHA, gmc_V)
  print(string.format("Gradient Monte Carlo, Episodes: %d, Time: %.2f", EPISODES, GetTime()))
  
  ResetTime()
  math.randomseed(0)
  local sgtd0_V = StateAggregationValueFunction:new{size = GROUPS, group_size = GROUP_SIZE}
  local distribution_td0 = SemiGradientTD0(ALPHA, sgtd0_V)
  print(string.format("Semi Gradient TD(0), Episodes: %d, Time: %.2f", EPISODES, GetTime()))
  
  local func_true_V = {color = RGB_GREEN}
  local func_gmc_V = {color = RGB_RED}
  local func_sgtd0_V = {color = RGB_RED}
  local func_distribution_mc = {color = RGB_WHITE}
  local func_distribution_td0 = {color = RGB_WHITE}
  for s = 1, STATES do
    func_true_V[s] = {x = s, y = true_V[s]}
    func_gmc_V[s] = {x = s, y = gmc_V:GetValue(s)}
    func_sgtd0_V[s] = {x = s, y = sgtd0_V:GetValue(s)}
    func_distribution_mc[s] = {x = s, y = distribution_mc[s]}
    func_distribution_td0[s] = {x = s, y = distribution_td0[s]}
  end
  graphs.funcs["True Value"] = func_true_V
  graphs.funcs["Aprroximate MC Value"] = func_gmc_V
  graphs_right.funcs["Distribution"] = func_distribution_mc

  local bmp = Bitmap.new(IMAGE_WIDTH, IMAGE_HEIGHT, RGB_BLACK)
  DrawGraphs(bmp, graphs, {int_x = true, skip_KP = true, center_x = 0, center_y = REWARD_LEFT})
  DrawGraphs(bmp, graphs_right, {int_x = true, skip_KP = true, center_x = 0, right_axis_Y = true, axis_y_format = "%.4f", bars = true})
  bmp:WriteBMP(IMAGE_FILENAME_MC)

  graphs.funcs["Aprroximate MC Value"] = nil
  graphs.funcs["Aprroximate TD(0) Value"] = func_sgtd0_V
  graphs_right.funcs["Distribution"] = func_distribution_td0
  local bmp = Bitmap.new(IMAGE_WIDTH, IMAGE_HEIGHT, RGB_BLACK)
  DrawGraphs(bmp, graphs, {int_x = true, skip_KP = true, center_x = 0, center_y = REWARD_LEFT})
  DrawGraphs(bmp, graphs_right, {int_x = true, skip_KP = true, center_x = 0, right_axis_Y = true, axis_y_format = "%.4f", bars = true})
  bmp:WriteBMP(IMAGE_FILENAME_TD0)
end

local function CreateTDnComparisonGraph(true_V)
  local graphs = {funcs = {}, name_x = "Alpha", name_y = string.format("Average RMS over %d states and first %d episodes", STATES, FIRST_EPISODES)}
  local colors = {RGB_RED, RGB_GREEN, RGB_BLUE, RGB_GRAY, RGB_MAGENTA, RGB_CYAN, RGB_WHITE, RGB_YELLOW, RGB_ORANGE, RGB_BROWN}
  local n, color_n = 1, 1
  while n <= MAX_N do
    ResetTime()
    local func = {color = colors[color_n], sort_idx = n}
    for alpha = 0.0, 1.0, ALPHA_STEP do
      local total_rms = 0.0
      math.randomseed(0)
      for run = 1, RUNS do
        local sgTDn_V = StateAggregationValueFunction:new{size = GROUPS_TDn, group_size = GROUP_SIZE_TDn}
        local rms = SemiGradientTDn(alpha, n, 1.0, sgTDn_V, FIRST_EPISODES, true_V)
        total_rms = total_rms + rms
      end
      total_rms = total_rms / (FIRST_EPISODES * RUNS)
      table.insert(func, {x = alpha, y = Clamp(total_rms, CLAMP_MIN_RMS, CLAMP_MAX_RMS)})
    end
    graphs.funcs[string.format("n=%d", n)] = func
    print(string.format("TD(%d) for %.2fs", n, GetTime()))
    n, color_n = n * 2, (color_n < #colors) and (color_n + 1) or 1
  end

  local bmp = Bitmap.new(IMAGE_WIDTH, IMAGE_HEIGHT, RGB_BLACK)
  DrawGraphs(bmp, graphs, {sort_cmp = function(a, b) return a.sort_idx < b.sort_idx end, center_y = CLAMP_MIN_RMS})
  bmp:WriteBMP(IMAGE_FILENAME_TDn)
end

local function AddBasisGraph(graphs, basis, order, alpha, true_V, color, sort_idx, name)
  math.randomseed(0)
  local rms = {}
  for run = 1, BASIS_RUNS do
    local value_function = BasisValueFunction:new{size = order, basis_type = basis}
    GradientMonteCarlo(alpha, value_function, BASIS_EPISODES, rms, true_V)
  end
  NormalizeRMS(rms, BASIS_RUNS)
  local func = {color = color, sort_idx = sort_idx}
  for episode, total_rms in ipairs(rms) do
    func[episode] = {x = episode, y = Clamp(rms[episode], BASIS_CLAMP_MIN_RMS, BASIS_CLAMP_MAX_RMS)}
  end
  graphs.funcs[name] = func
end

local function CreatePolynomialFourierBasisComparison(true_V)
  local graphs = {funcs = {}, name_x = "Episodes", name_y = string.format("Average RMS over %d states and first %d episodes", STATES, BASIS_EPISODES)}
  local colors = {RGB_RED, RGB_GREEN, RGB_BLUE, RGB_MAGENTA, RGB_CYAN, RGB_WHITE}
  local color_n = 1
  
  -- polynomials
  for _, order in ipairs(BASIS_ORDERS) do
    ResetTime()
    AddBasisGraph(graphs, "polynomials", order, ALPHA_POLYNOMIAL, true_V, colors[color_n], color_n, string.format("P order=%d", order))
    print(string.format("MC Polynomial order-%d for %.2fs", order, GetTime()))
    color_n = (color_n < #colors) and (color_n + 1) or 1
  end
  
  -- Fourier
  for _, order in ipairs(BASIS_ORDERS) do
    ResetTime()
    AddBasisGraph(graphs, "fourier", order, ALPHA_FOURIER, true_V, colors[color_n], color_n, string.format("F order=%d", order))
    print(string.format("MC Fourier order-%d for %.2fs", order, GetTime()))
    color_n = (color_n < #colors) and (color_n + 1) or 1
  end

  local bmp = Bitmap.new(IMAGE_WIDTH, IMAGE_HEIGHT, RGB_BLACK)
  DrawGraphs(bmp, graphs, {skip_KP = true, int_x = true, sort_cmp = function(a, b) return a.sort_idx < b.sort_idx end, center_y = BASIS_CLAMP_MIN_RMS})
  bmp:WriteBMP(IMAGE_FILENAME_BASIS)
end

local function AddTilingGraph(graphs, tilings, true_V, color, name)
  math.randomseed(0)
  local tile_offset = (tilings > 1) and TILE_OFFSET or 0
  local rms = {}
  for run = 1, TILING_RUNS do
    local Tiling = TileCoding:new{size = tilings, tile_width = TILE_WIDTH, tile_offset = tile_offset}
    GradientMonteCarlo(ALPHA_TILINGS / tilings, Tiling, TILING_EPISODES, rms, true_V)
  end
  NormalizeRMS(rms, TILING_RUNS)
  local func_tiling = {color = color}
  for episode, rms_ep in ipairs(rms) do
    func_tiling[episode] = {x = episode, y = Clamp(rms_ep, TILING_CLAMP_MIN_RMS, TILING_CLAMP_MAX_RMS)}
  end
  graphs.funcs[name] = func_tiling
end

local function CreateTilingGraph(true_V)
  local graphs = {funcs = {}, name_x = "Episodes", name_y = string.format("Average RMS over %d states and first %d episodes", STATES, TILING_EPISODES)}
  ResetTime()
  AddTilingGraph(graphs, TILINGS, true_V, RGB_GREEN, string.format("%d-Tiling", TILINGS))
  print(string.format("%d-Tiling Monte Carlo, Episodes: %d, Time: %.2f", TILINGS, TILING_EPISODES, GetTime()))
  ResetTime()
  AddTilingGraph(graphs, 1, true_V, RGB_RED, "No Tiling")
  print(string.format("Tiling-1 Monte Carlo, Episodes: %d, Time: %.2f", TILING_EPISODES, GetTime()))
  local bmp = Bitmap.new(IMAGE_WIDTH, IMAGE_HEIGHT, RGB_BLACK)
  DrawGraphs(bmp, graphs, {skip_KP = true, int_x = true, center_x = 0, center_y = TILING_CLAMP_MIN_RMS})
  bmp:WriteBMP(IMAGE_FILENAME_TILING)
end

local true_V = ComputeTrueValue()

CreateMCTD0Graphs(true_V)
CreateTDnComparisonGraph(true_V)
CreatePolynomialFourierBasisComparison(true_V)
CreateTilingGraph(true_V)
