# -*- coding: utf-8 -*-

"""utils."""

import logging

from almost_unique_id import generate_id


logger = logging.getLogger(__name__)


def get_id_func():
    id = generate_id()

    def get_id():
        return id

    return get_id
