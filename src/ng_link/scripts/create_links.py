import dataset_links

if __name__ == '__main__':
    # Fill in your own data
    base_channel_path = '/Users/jonathan.wong/Projects/aind-ng-link/dispim_new.xml'
    cross_channel_path = '/Users/jonathan.wong/Projects/aind-ng-link/dispim_new_coreg.xml'
    s3_path = "s3://aind-open-data/diSPIM_647459_2022-12-21_00-39-00/diSPIM.zarr"
    
    dataset_links.generate_exaspim_link(base_channel_path, 
                                        s3_path, 
                                        max_dr = 200, 
                                        opacity = 1.0, 
                                        blend = "default",
                                        output_json_path = ".")
    # dataset_links.generate_dispim_link(base_channel_path, 
    #                                    cross_channel_path, 
    #                                    s3_path, 
    #                                    max_dr = 800, 
    #                                    opacity = 0.5, 
    #                                    blend = "additive",
    #                                    deskew_angle = 45, 
    #                                    output_json_path = ".")