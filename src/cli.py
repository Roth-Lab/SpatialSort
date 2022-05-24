from src.inference.run import run
import click


@click.command()
@click.option("--exp-csv", required=True, type=click.Path(exists=True, resolve_path=True),
              help="The file path of the csv file holding expression data. The first column is file_id which is the patient id.")
@click.option("--loc-csv", required=True, type=click.Path(exists=True, resolve_path=True),
              help="The file path of the csv file holding location data. The first column is file_id which is the patient id.")
@click.option("--rel-csv", required=True, type=click.Path(exists=True, resolve_path=True),
              help="The file path of the csv file holding cell neighbour relations data. The first column is file_id which is the patient id.")
@click.option("-k", "--num-clusters", required=True, type=int, help="Number of clusters to initialize clustering")
@click.option("-o", "--out_dir", required=True, type=str, help="Output directory.")
@click.option("-t", "--num-iters", default=500, type=int, help="Number of iterations to run inference.")
@click.option("-s", "--seed", default=None, type=int, help="A seed for random numbers. Default is a random seed.")
@click.option("-p", "--prior-csv", default=None, type=click.Path(exists=True, resolve_path=True),
              help="The file path of the csv file holding a cluster by gene prior expression matrix with entries in [-1,0,1] for low, mid, hi values.")
@click.option("-a", "--anchor-csv", default=None, type=click.Path(exists=True, resolve_path=True),
              help="The file path of the csv file holding anchor data for each known cluster.")
@click.option("-l", "--prec-scale", default=0.1, type=float, help="Precision scale parameter, smaller the stronger.")
@click.option("--save-trace/--no-save-trace", default=False, help="Whether to save the label trace.")
def infer(**kwargs):
    """Performing inference for SpatialSort."""
    run(**kwargs)


@click.group(name='SpatialSort')
def main():
    pass


main.add_command(infer)

if __name__ == '__main__':
    main()
