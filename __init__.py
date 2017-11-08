# -*- coding: utf-8 -*-
"""
/***************************************************************************
 HandGestures
                                 A QGIS plugin
 Zoom and pan using hand gestures
                             -------------------
        begin                : 2016-10-29
        copyright            : (C) 2016 by Sebastian Roll
        email                : sebastianroll84@gmail.com
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load HandGestures class from file HandGestures.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .hand_gestures import HandGestures
    return HandGestures(iface)
