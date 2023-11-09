import argparse
import dispim_link

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xml_in",
        type=str,
        help="Bigstitcher XML to make Neuroglancer link of.")
    parser.add_argument("s3_bucket", type=str, help="S3 bucket name.",
                        default='aind-open-data')

    args = parser.parse_args()

    # print(args)
    xml_file_in = args.xml_in

    s3_bucket = args.s3_bucket

    ng_link = dispim_link.ingest_xml_and_write_ng_link(xml_file_in, s3_bucket)

    print(ng_link)
