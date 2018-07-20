"""Command line interface for tensionflow"""
import click


@click.command()
@click.option('-v', '--verbose', count=True)
def cli(verbose):
    click.echo(click.style('Hello tensionflow!', fg='magenta', bold=True))
    click.echo('Verbosity level: {}'.format(verbose))
