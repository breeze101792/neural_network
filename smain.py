#!/usr/bin/env python3
from gi.repository import Gtk, Gdk
from gi.repository.GdkPixbuf import Pixbuf
import matplotlib.pyplot as plt
import threading
from perceptron import *
import numpy as np
from data_proc import data_proc as dp
from ploting import paper
from matplotlib import colors
import settings

from som import SOM

# color = ["r","b","g","c","m","y","k","w"]

class myDict(dict):

    def __init__(self):
        self = dict()

    def add(self, key, value):
        self[key] = value


UI_INFO = """
<ui>
  <menubar name='MenuBar'>
    <menu action='FileMenu'>
      <!--menu action='FileNew'>
        <menuitem action='FileNewStandard' />
        <menuitem action='FileNewFoo' />
        <menuitem action='FileNewGoo' />
      </menu-->
      <menuitem action='FileOpen' />
      <menuitem action='FileSave' />
      <separator />
      <menuitem action='FileQuit' />
    </menu>
    <menu action='AboutMenu'>
      <menuitem action='About'/>
    </menu>
  </menubar>
  <toolbar name='ToolBar'>
    <!--toolitem action='FileNewStandard' /-->
    <toolitem action='FileOpen' />
    <!--toolitem action='FileSave' /-->
    <toolitem action='Run' />
    <toolitem action='FileQuit' />
  </toolbar>
  <!--popup name='PopupMenu'>
    <menuitem action='EditCopy' />
    <menuitem action='EditPaste' />
    <menuitem action='EditSomething' />
  </popup-->
</ui>
"""
class info:
    class traning_data:
        def __init__(self):
            self.Iteration_times = 0
            self.Error_rate = 0
            self.Data_set_size = 0
        def reset(self):
            self.Iteration_times = 0
            self.Error_rate = 0
            self.Data_set_size = 0
    class testing_data:
        def __init__(self):
            self.Error_rate = 0
            self.Data_set_size = 0
        def reset(self):
            self.Error_rate = 0
            self.Data_set_size = 0
    def __init__(self):
        self.traning = self.traning_data()
        self.testing = self.testing_data()
    def reset(self):
        self.traning.reset()
        self.testing.reset()

class nNetwork(Gtk.Window):
    def __init__(self):
        self.nnetwork = None
        self.dimension = 2
        self.train_mode = True
        #self.data = []
        self.weights = []
        self.class_table = {}
        self.find_best = False
        #self.class_num
        self.dataset = dp()
        self.training_set = []
        self.testing_set = []
        self.traning_trainsformed_data = []
        self.testing_trainsformed_data = []
        #log info
        self.nninfo = info()

        #wait for ui setup
        Gtk.Window.__init__(self, title="Neural Network")
        self.set_default_size(800, 700)

        action_group = Gtk.ActionGroup("my_actions")

        #menu item
        self.add_file_menu_actions(action_group)
        self.add_about_menu_actions(action_group)

        #create ui manager
        uimanager = self.create_ui_manager()
        uimanager.insert_action_group(action_group)

        menubar = uimanager.get_widget("/MenuBar")

        main_ui = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(main_ui)
        main_ui.pack_start(menubar, False, False, 0)

        toolbar = uimanager.get_widget("/ToolBar")
        main_ui.pack_start(toolbar, False, False, 0)

        # body pannel settings
        body_panel = Gtk.Table(2, 3, True)
        main_ui.pack_start(body_panel, True, True, 0)


        # *********************************************************************/
        #drawing
        self.traning_draw_draw_panel = Gtk.Box(10, 2, True)
        body_panel.attach(self.traning_draw_draw_panel, 0, 3, 0, 2)

        self.traning_draw_paper = paper(title="SOM")
        self.traning_draw_draw_panel.pack_start(self.traning_draw_paper.canvas, True, True, 0)
        self.traning_draw_paper.resetpaper()

        self.training_set = None


        # self.classifier = None
    def add_file_menu_actions(self, action_group):
        action_filemenu = Gtk.Action("FileMenu", "File", None, None)
        action_group.add_action(action_filemenu)

        action_filenewmenu = Gtk.Action("FileNew", None, None, Gtk.STOCK_NEW)
        action_group.add_action(action_filenewmenu)

        action_new = Gtk.Action("FileNewStandard", "_New",
            "Create a new file", Gtk.STOCK_NEW)
        action_new.connect("activate", self.on_menu_file_new_generic)
        action_group.add_action_with_accel(action_new, None)

        action_group.add_actions([
            ("FileNewFoo", None, "New Foo", None, "Create new foo",
             self.on_menu_file_new_generic),
            ("FileNewGoo", None, "_New Goo", None, "Create new goo",
             self.on_menu_file_new_generic),
        ])

        action_fileopen = Gtk.Action("FileOpen", "Open", None, Gtk.STOCK_OPEN)
        action_fileopen.connect("activate", self.on_menu_file_open)
        action_group.add_action(action_fileopen)

        action_filesave = Gtk.Action("FileSave", "Save", None, Gtk.STOCK_SAVE)
        action_group.add_action(action_filesave)

        action_run = Gtk.Action("Run", "Run", None, Gtk.STOCK_MEDIA_PLAY)
        action_run.connect("activate", self.on_clicked_run)
        action_group.add_action(action_run)

        action_filequit = Gtk.Action("FileQuit", None, None, Gtk.STOCK_QUIT)
        action_filequit.connect("activate", self.on_menu_file_quit)
        action_group.add_action(action_filequit)

    def add_about_menu_actions(self, action_group):
        action_aboutmenu = action_group.add_action(Gtk.Action("AboutMenu", "About", None,
            None))
        action_aboutmenu = Gtk.Action("About", "About", None, self.on_menu_others)
        action_group.add_action(action_aboutmenu)

    def create_ui_manager(self):
        uimanager = Gtk.UIManager()

        # Throws exception if something went wrong
        uimanager.add_ui_from_string(UI_INFO)

        # Add the accelerator group to the toplevel window
        accelgroup = uimanager.get_accel_group()
        self.add_accel_group(accelgroup)
        return uimanager

    def on_menu_file_new_generic(self, widget):
        print("A File|New menu item was selected.")

    def on_menu_file_quit(self, widget):
        Gtk.main_quit()

    def on_menu_file_open(self, widget):
        dialog = Gtk.FileChooserDialog("Please choose a file", self,
            Gtk.FileChooserAction.OPEN,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
             Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        self.add_filters(dialog)

        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            #reset
            # self.nninfo.reset()
            # self.log_refresh()

            print("Open clicked")
            print("File selected: " + dialog.get_filename())
            # print(self.traning_testing_rate_sb.get_value() / 100)
            self.dataset.set_file_name(dialog.get_filename())
            self.dataset.open_file()
            self.training_set = self.dataset.get_data()
            #
            # #log info
            # file_name = dialog.get_filename().split('/')[-1]
            #
            #
            # if self.dataset.get_data_size() > 100:
            #     print("tt rate\t", self.traning_testing_rate_sb.get_value())
            #     self.training_set = self.dataset.get_data(self.traning_testing_rate_sb.get_value() / 100)
            #     self.training_set = (dp.to_ndata(self.training_set[0]), self.training_set[1])
            #     self.testing_set = self.dataset.get_data(1 - self.traning_testing_rate_sb.get_value() / 100)
            #     self.testing_set = (dp.to_ndata(self.testing_set[0]), self.testing_set[1])
            # else:
            #     self.training_set = self.dataset.get_data(1)
            #     self.training_set = (dp.to_ndata(self.training_set[0]), self.training_set[1])
            #     self.testing_set = self.dataset.get_data(1)
            #     self.testing_set = (dp.to_ndata(self.testing_set[0]), self.testing_set[1])
            # self.nninfo.traning.Data_set_size = len(self.training_set[1])
            # self.nninfo.testing.Data_set_size = len(self.testing_set[1])
            #
            # if self.dataset.get_data_dimension() == 2:
            #     self.ori_paper.resetpaper()
            #     self.ori_paper.draw_2d_point(self.dataset.get_data(1, is_random = True),self.dataset.get_class_middle())
            #     self.ori_paper.expend_lim()
            #     self.ori_paper.draw()
            # else:
            #     self.ori_paper.resetpaper()
            #     self.ori_paper.draw()
            #
            # self.dataset_info_msg_lab.set_text((file_name[0:18] + "..." if len(file_name) > 20 else file_name) + " \n" + self.dataset.get_data_size().__str__() + " \n" + self.dataset.get_data_dimension().__str__() + " \n" + self.dataset.get_data_classification_num().__str__() + " ")
            #
            # self.traning_draw_paper.resetpaper()
            # self.testing_draw_paper.resetpaper()
        elif response == Gtk.ResponseType.CANCEL:
            print("Cancel clicked")
        #print(self.data, ", ", self.class_table, ", ", tmp_table)
        dialog.destroy()
    def add_filters(self, dialog):
        filter_text = Gtk.FileFilter()
        filter_text.set_name("Text files")
        filter_text.add_mime_type("text/plain")
        dialog.add_filter(filter_text)

        filter_py = Gtk.FileFilter()
        filter_py.set_name("Python files")
        filter_py.add_mime_type("text/x-python")
        dialog.add_filter(filter_py)

        filter_any = Gtk.FileFilter()
        filter_any.set_name("Any files")
        filter_any.add_pattern("*")
        dialog.add_filter(filter_any)
    def on_clicked_train(self, widget):
        x = 25
        y = 25
        print("train")
        som = SOM(x,y)
        som.net_init()
        som.training(self.training_set[0])
        # som.printnet()
        self.traning_draw_paper.resetpaper()
        self.traning_draw_paper.draw_net(som.get_net(), x,y)
        self.traning_draw_draw_panel.queue_draw()
        print("end")
        self.traning_draw_paper.draw_2d_point(self.training_set, self.dataset.get_class_middle())
    def on_clicked_run(self, widget):

        self.on_clicked_train(widget)


    def on_menu_others(self, widget):
        print("Menu item " + widget.get_name() + " was selected")

def main():
    window = nNetwork()
    window.connect("delete-event", Gtk.main_quit)
    window.show_all()
    Gtk.main()

if __name__ == '__main__':
    main()