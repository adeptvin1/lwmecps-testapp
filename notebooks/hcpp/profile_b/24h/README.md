generate_base_stations(fixed_seed=97)
generate_users(hcpp_network.num_users, distribution='uniform', fixed_seed=14)
plot_users_per_station_over_time(hcpp_network, load_dist, base_stations, fixed_seed=14)