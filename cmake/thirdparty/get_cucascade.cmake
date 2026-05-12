# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# We build cuCascade as a static library to avoid packaging issues with wheels.
function(find_and_configure_cucascade)
  rapids_cpm_find(
    cuCascade 0.1.0
    GLOBAL_TARGETS cuCascade::cucascade
    CPM_ARGS
    GIT_REPOSITORY https://github.com/NVIDIA/cuCascade.git
    GIT_TAG main
    GIT_SHALLOW TRUE
    OPTIONS "CUCASCADE_BUILD_TESTS OFF"
            "CUCASCADE_BUILD_BENCHMARKS OFF"
            "CUCASCADE_BUILD_SHARED_LIBS OFF"
            "CUCASCADE_BUILD_STATIC_LIBS ON"
            "CUCASCADE_WARNINGS_AS_ERRORS OFF"
    EXCLUDE_FROM_ALL
  )

  # Create an interface library that wraps cuCascade to avoid export conflicts. This target won't be
  # exported but can be used internally. Link cuDF transitive dependencies explicitly.
  if(TARGET cuCascade::cucascade AND NOT TARGET rapidsmpf_cucascade_internal)
    add_library(rapidsmpf_cucascade_internal INTERFACE)
    target_link_libraries(rapidsmpf_cucascade_internal INTERFACE cuCascade::cucascade)
    set_target_properties(rapidsmpf_cucascade_internal PROPERTIES EXPORT_NAME "")
  endif()
endfunction()

find_and_configure_cucascade()
