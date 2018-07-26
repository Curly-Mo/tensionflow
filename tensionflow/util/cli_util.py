import logging

import click

logger = logging.getLogger(__name__)


class Mutex(click.Option):
    def __init__(self, *args, **kwargs):
        self.mutex_with = kwargs.pop('mutex_with')
        if self.mutex_with is None:
            raise click.UsageError('`mutex_with` parameter required')
        kwargs[
            'help'
        ] = f'{kwargs.get("help", "")}Option is mutually exclusive with {", ".join(self.mutex_with)}.'.strip()
        super(Mutex, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if not any(opt in opts for opt in self.mutex_with + [self.name]):
            raise click.UsageError(f'Usage error: One of {self.mutex_with + [self.name]} is required.')
        for mutex_opt in self.mutex_with:
            if mutex_opt in opts:
                if self.name in opts:
                    raise click.UsageError(
                        f'Usage error: `{str(self.name)}` is mutually exclusive with `{str(mutex_opt)}`.'
                    )
        return super(Mutex, self).handle_parse_result(ctx, opts, args)
