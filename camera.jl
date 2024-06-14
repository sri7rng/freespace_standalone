using Images, MLUtils, Tullio, Logging, GenericRofReader

# camera
luma_interface_to_data(interface) =
  interface.m_buffer.m_inheritedStructure.m_value.data

chroma_interface_to_data(interface) =
  interface.m_buffer.m_inheritedStructure.m_value.data
chroma_pixel_to_value(pixel) = pixel.m_value

camera_interface_to_timestamp(interface) =
  interface.m_globalTimestamp.m_value.m_value
camera_interface_to_camera_id(interface) = interface.m_camId
camera_interface_to_image_hash(interface) = interface.m_imageHash

function interfaces()
  return Dict(
    :nrcs_front => Dict(
      :camera_luma => "8449",
      :camera_chroma => "8450",
    ),
    :nrcs_rear => Dict(
      :camera_luma => "8455",
      :camera_chroma => "8482",
    ),
    :nrcs_left => Dict(
      :camera_luma => "8454",
      :camera_chroma => "8481",
    ),
    :nrcs_right => Dict(
      :camera_luma => "8456",
      :camera_chroma => "8483",
    ),
  )
end

const interface_ids = interfaces()

# ultrasound
#

function load_interface_samples(
  path_to_rof,
  dataset_type::Symbol,
  interface_type::Symbol,
)
  # shut noisy GenericRofReader up
  quiet_logger = SimpleLogger(stdout, Logging.Error)
  original_logger = global_logger()
  global_logger(quiet_logger)

  # read appropriate interface from rof file
  rof_ids = get_unique_source_ids(path_to_rof)
  i = findfirst(x -> x == interface_ids[dataset_type][interface_type], rof_ids)
  id = rof_ids[i]
  namespace = Module(:UniqueNameSpace)
  reader = initialize_reader(path_to_rof, id, namespace)
  interface_samples = read_all_valid_interfaces(reader)
  global_logger(original_logger)

  return interface_samples
end

"""
Camera data.

# Fields
- `image::Matrix{RGB{N0f8}}`: The RGB image generated from the luma and chroma values.
- `timestamp::UInt32`: The timestamp of the image (unit currently unknown).
- `camera_id::Int32`: The ID of the camera.
"""
@kwdef struct CameraData
  image::Array{Float64}
  timestamp::UInt32
  camera_id::Int32
end

# this matrix transforms YCbCr (ITU-R BT.2020) to RGB
const trafo = [
  1 0 1.402
  1 -0.344136 -0.714136
  1 1.772 0
]
const umax = 0.436
const vmax = 0.615

"""
    split_uv(uv::UInt16)::Tuple{UInt8,UInt8}

Split the 16 bit UV value into two 8 bit values.

# Arguments
- `uv::UInt16`: The 16 bit UV value.

# Returns
- `Tuple{UInt8,UInt8}`: The 8 bit U and V values.
"""
function split_uv(uv::UInt16)::Tuple{UInt8,UInt8}
  # The 8 least significant bits are the V value
  u = uv & 0xff
  # The 8 most significant bits are the V value
  v = uv >> 8
  return u, v
end

"""
    uv_to_u(uv::UInt16)::UInt8

Extract the U value from the 16 bit UV value.
"""
function uv_to_u(uv::UInt16)::UInt8
  return uv & 0xff
end

function uv_to_v(uv::UInt16)::UInt8
  return uv >> 8
end

"""
    extract_luma(rof_luma_data::NTuple{517120,UInt16})::Matrix{UInt16}

Extract the luma data from the ROF file.

# Arguments
- `rof_luma_data::NTuple{517120,UInt16}`: The luma data from the ROF file.

# Returns
- `Matrix{Float64}`: The luma matrix.
"""
function extract_luma(rof_luma_data::NTuple{517120,UInt16})::Matrix{Float64}
  luma_vector = collect(rof_luma_data)
  #luma_vector = (luma_vector .>>> 4) .& 0x00FF
  #println("1. extract_luma min:", minimum(luma_vector), " max:", maximum(luma_vector), " shape:", size(luma_vector))
  luma_matrix = reshape(luma_vector, 808, 640) .|> scale_y
  #println("2. extract_luma min:", minimum(luma_matrix), " max:", maximum(luma_matrix), " shape:", size(luma_vector))
  return luma_matrix
end

"""
    extract_chroma(
      rof_chroma_data::NTuple{129280,<:Any},
    )

Extract the chroma data from the ROF file.

# Arguments
- `rof_chroma_data::NTuple{129280,<:Any}`: The chroma data from the ROF file.

# Returns
- `Tuple{Matrix{Float64},Matrix{Float64}}`: The U and V matrices.
"""
function extract_chroma(
  rof_chroma_data::NTuple{129280,<:Any},
)::Tuple{Matrix{Float64},Matrix{Float64}}
  chroma_vector = map(rof_chroma_data) do pixel
    chroma_pixel_to_value(pixel)
  end |> collect
  chroma_matrix = reshape(chroma_vector, 404, 320)
  # repeat due to lower resolution of chroma compared to luma
  chroma_matrix = repeat(chroma_matrix, inner = (2, 2))
  #println("1. extract_chroma min:", minimum(chroma_matrix), " max:", maximum(chroma_matrix), " shape:", size(chroma_matrix))
  u_matrix = uv_to_u.(chroma_matrix) .|> scale_u
  v_matrix = uv_to_v.(chroma_matrix) .|> scale_v
  #println("2. extract_chroma U_matrix min:", minimum(u_matrix), " max:", maximum(u_matrix), " shape:", size(u_matrix))
  #println("2. extract_chroma V_matrix min:", minimum(v_matrix), " max:", maximum(v_matrix), " shape:", size(v_matrix))
  return u_matrix, v_matrix
end

"""
    scale_yuv(y::UInt16, u::UInt8, v::UInt8)

Scale the YUV values to the range [0, 1], [-umax, umax], and [-vmax, vmax].

# Arguments
- `y::UInt16`: The Y value.
- `u::UInt8`: The U value.
- `v::UInt8`: The V value.

# Returns
- `Tuple{Float64,Float64,Float64}`: The scaled Y, U, and V values.
"""
function scale_yuv(y::UInt16, u::UInt8, v::UInt8)
  yscaled = scale_y(y)
  uscaled = scale_u(u)
  vscaled = scale_v(v)
  return yscaled, uscaled, vscaled
end

const uv_delta = 0.5

# scale of u to [-umax, umax]
scale_u(u::UInt8) = (u / 0xff) - uv_delta # * 2 * umax - umax
# scale of v to [-vmax, vmax]
scale_v(v::UInt8) = (v / 0xff) - uv_delta # * 2 * vmax - vmax
# unit scale of y
scale_y(y::UInt16) = clamp(y, 0x0, 0xfff) / 0xfff

"""
    rgb_image(
      luma::Matrix{UInt16},
      chroma::Matrix{Tuple{UInt8,UInt8}},
    )::Matrix{RGB{N0f8}}

Assemble the luma and chroma data into an RGB image.

# Arguments
- `luma::Matrix{UInt16}`: The luma data.
- `chroma::Matrix{Tuple{UInt8,UInt8}}`: The chroma data.

# Returns
- `Array{Float32}`: The RGB image.
"""
function rgb_image(
  luma::Matrix{Float64},
  u_matrix::Matrix{Float64},
  v_matrix::Matrix{Float64},
)

  yuv_image = stack((luma, u_matrix, v_matrix), dims = 1)
  @tullio float_image[channel, width, height] :=
    trafo[channel, channel2] * yuv_image[channel2, width, height]
  image = permutedims(clamp!(float_image*255.0, 0.0, 255.0), (3, 2, 1)) .|> Float32
  return image #[H, W, C]
end


"""
    convert_rof_to_camera_data(
      rof_path::AbstractString,
      max_images::Union{Int,Nothing} = nothing
    )::Vector{CameraData}

Convert a ROF file to an RGB image.

# Arguments
- `rof_path::AbstractString`: The path of the NRCS ROF file.
- `img_idx::Integer`: Index from which on image reading starts
- `batch_size::Integer`: Number of images that shall be read and returned
- `camera_str::AbstractString`: String of which camera shall be read (front,left,rear,right)
- `max_images::Union{Int,Nothing} = nothing`: The maximum number of images to convert. Optional.

# Returns
- `Vector{CameraData}`: The camera data for every sample.
"""
function convert_rof_to_camera_data(
  rof_path::AbstractString,
  img_idx::Integer,
  batch_size::Integer,
  camera_str::AbstractString,
  max_images::Union{Int,Nothing} = nothing,
)
  println("JULIA | Reading ", batch_size, " images at index:", img_idx)
  # Load data.
  if (camera_str == "front")
    dataset_type = :nrcs_front
  elseif (camera_str == "left")
    dataset_type = :nrcs_left
  elseif (camera_str == "right")
    dataset_type = :nrcs_right
  else
    dataset_type = :nrcs_rear
  end
  luma_sample = load_interface_samples(rof_path, dataset_type, :camera_luma)
  chroma_sample = load_interface_samples(rof_path, dataset_type, :camera_chroma)

  # Create a dictionary to map image number to indices.
  # This is necessary in order to associate luma / chroma pairs together.
  # Due to the nature of the data, samples may be missing at the beginning, the middle and / or the end.
  # The missing samples may also be different for luma / chroma.
  # Thus, we associate luma / chroma samples together by their image number.
  # Time stamp can not be used directly since it overlflows leading to wrong order when sorting.
  luma_ts_idx_mapping = Dict(
    sample.m_inheritedStructure.m_inheritedStructure.m_imageNumber => [i, sample.m_globalTimestamp.m_value.m_value] for
    (i, sample) in enumerate(luma_sample)
  )
  chroma_ts_idx_mapping = Dict(
    sample.m_inheritedStructure.m_inheritedStructure.m_imageNumber => [i, sample.m_globalTimestamp.m_value.m_value] for
    (i, sample) in enumerate(chroma_sample)
  )

  # Find the intersection of the timestamps and sort them.
  timestamps =
    intersect(keys(luma_ts_idx_mapping), keys(chroma_ts_idx_mapping)) |>
    collect |>
    sort
  # timestamps =
  #   intersect(keys(luma_ts_idx_mapping), keys(chroma_ts_idx_mapping)) |>
  #   collect

    @assert length(timestamps) != 0
  # Check if luma and chroma are matching in pairs.
  if length(timestamps) != length(luma_sample) ||
     length(timestamps) != length(chroma_sample)
    @warn "Warning: Luma and Chroma collections have different lengths. Truncated to the length of cooccuring timestamps. Luma length: $(length(luma_sample)), Chroma length: $(length(chroma_sample))"
  end

  # Limit the number of images if necessary.
  if !isnothing(max_images)
    timestamps = timestamps[1:min(max_images, length(timestamps))]
  end

  #timestamps_str = [string(stamp, base=16) for stamp in timestamps]
  #println("JULIA | timestamps: ", length(timestamps_str), timestamps_str)
  #println("JULIA | timestamps: ", length(keys(chroma_ts_idx_mapping)), keys(chroma_ts_idx_mapping))

  # Get the respective luma and chroma indices of the timestamps.
  luma_idx = [luma_ts_idx_mapping[timestamp][1] for timestamp in timestamps]
  chroma_idx = [chroma_ts_idx_mapping[timestamp][1] for timestamp in timestamps]
  m_global_timestamps = [luma_ts_idx_mapping[timestamp][2] for timestamp in timestamps]

  #println("JULIA | Before chroma_sample idx: ", length(chroma_sample))
  # Get the luma and chroma samples.
  luma_sample = getindex(luma_sample, luma_idx)
  chroma_sample = getindex(chroma_sample, chroma_idx)

  # println("JULIA | After chroma_sample idx: ", length(chroma_sample))
  # println("JULIA | Luma idx: ", length(luma_idx), luma_idx)
  # println("JULIA | Chroma idx: ", length(chroma_idx), chroma_idx)

  # Initialize empty array.
  result_length = min(length(luma_sample), img_idx+batch_size) - img_idx
  camera_data_array = Array{Array{Float64}}(undef, result_length)
  #println("JULIA | Cam data array: ", length(camera_data_array), "Result length: ", result_length)
  range_idxs = range(img_idx, step=1, length=result_length)#img_idx is 0 indexed
  for i in range_idxs
    if (i%250==0)
      println("JULIA | Reading samples at index: ", i, "/",length(luma_sample))
    end
    camera_data_array[(i%batch_size)+1] =
      convert_sample_to_camera_data(luma_sample[i+1], chroma_sample[i+1])
  end
  return camera_data_array, m_global_timestamps
end

"""
    convert_sample_to_camera_data(
      luma_sample,
      chroma_sample
    )::CameraData

Converts luma and chroma data to an RGB image.

# Arguments
- `luma_sample`: The luma data.
- `chroma_sample`: The chroma data.

# Returns
- `CameraData`: The camera data.
"""
function convert_sample_to_camera_data(luma_sample, chroma_sample)
  # Extract luma data.
  luma = luma_interface_to_data(luma_sample) |> extract_luma

  # Extract chroma data.
  u_matrix, v_matrix = chroma_interface_to_data(chroma_sample) |> extract_chroma

  # Extract metadata
  timestamp = camera_interface_to_timestamp(luma_sample)
  camera_id = camera_interface_to_camera_id(luma_sample)

  # Convert to RGB image.
  return rgb_image(luma, u_matrix, v_matrix)
end

"""
    check_same_image(luma_sample, chroma_sample)

Do luma and chroma samples belong to the same image?

# Arguments
- `luma_sample`: The luma sample.
- `chroma_sample`: The chroma sample.
"""
function check_same_image(luma_sample, chroma_sample)
  # Source for m_timestamp
  # Likely not the correct one as there are several NRCS repositories.
  # https://sourcecode01.de.bosch.com/projects/NRCS2CN/repos/cpj_gac_nrc2_us/browse/rtaos/pf/cv/imp/lad_lcam/inc/lad_lcam_image_meta_info_1_cam_output_v2.hpp#60,80
  # Using: "global timestamp (synchronized between all cores) of current image"
  # Note: m_timestamp is deprecated and m_localTimestamp is unsynchronized.
  @assert camera_interface_to_timestamp(luma_sample) ==
          camera_interface_to_timestamp(chroma_sample)
  @assert camera_interface_to_camera_id(luma_sample) ==
          camera_interface_to_camera_id(chroma_sample)
  @assert camera_interface_to_image_hash(luma_sample) ==
          camera_interface_to_image_hash(chroma_sample)
  # Possible source for m_freqIndicator.
  # https://sourcecode01.de.bosch.com/projects/NRCS2CN/repos/cpj_hryt_nrc2_us/browse/rtaos/pf/cv/imp/lad_lcam/inc/lad_lcam_tabs.hpp#112
  # However values 0x03, 0x07 do not match.
  # Not sure how useful this is.
end

#dat, ts = convert_rof_to_camera_data("/app/data/mf4_conversion/755573/20240222T085641Z_LBXO1994_C1_DEV_ARGUS_LiDAR_MTA2_0_Recording.rof", 0, 10, "front")
