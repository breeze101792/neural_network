#!/usr/bin/env python3
from gi.repository import Gtk, Gdk, GObject
from gi.repository.GdkPixbuf import Pixbuf
import matplotlib.pyplot as plt
import threading
from perceptron import *
import numpy as np
from data_proc import data_proc as dp
from ploting import paper
from racing_map import racing_map
from matplotlib import colors
from racing_map import *
from ga import rbfn
import time
import copy
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
        # self.class_num
        self.dataset = dp()
        self.training_set = []
        self.testing_set = []
        self.traning_trainsformed_data = []
        self.testing_trainsformed_data = []
        # log info
        self.nninfo = info()
        self.rbfn = None

        # wait for ui setup
        Gtk.Window.__init__(self, title="Neural Network")
        self.set_default_size(800, 700)
        # self.set_default_size(800, 700)
        # self.connect("key-press-event", self.on_window_key_press_event)

        action_group = Gtk.ActionGroup("my_actions")

        # menu item
        self.add_file_menu_actions(action_group)
        self.add_about_menu_actions(action_group)

        # create ui manager
        uimanager = self.create_ui_manager()
        uimanager.insert_action_group(action_group)

        menubar = uimanager.get_widget("/MenuBar")

        main_ui = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(main_ui)
        main_ui.pack_start(menubar, False, False, 0)

        toolbar = uimanager.get_widget("/ToolBar")
        main_ui.pack_start(toolbar, False, False, 0)

        # body pannel settings
        body_panel = Gtk.Table(2, 4, True)
        main_ui.pack_start(body_panel, True, True, 0)

        # for
        # Settings/*********************************************************/
        settings_panel = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=10)
        body_panel.attach(settings_panel, 0, 1, 0, 2, xpadding=8, ypadding=8)

        settings_lab = Gtk.Label("Settings", xalign=0)
        settings_panel.pack_start(settings_lab, False, False, 0)

        # data settings
        # *******************************************************/
        data_settings_lab = Gtk.Label("Data Settings", xalign=0)
        settings_panel.pack_start(data_settings_lab, False, False, 0)

        # traning_testing_rate_group
        traning_testing_rate_group = Gtk.Table(1, 2, True)
        settings_panel.pack_start(traning_testing_rate_group, False, False, 0)
        traning_testing_rate_lab = Gtk.Label(
            "Traning & Tesgting\ndata rate(%):", xalign=0)
        traning_testing_rate_group.attach(traning_testing_rate_lab, 0, 1, 0, 1)
        traning_testing_rate_adj = Gtk.Adjustment(60, 0, 100, 5, 0, 0)
        self.traning_testing_rate_sb = Gtk.SpinButton()
        self.traning_testing_rate_sb.set_alignment(xalign=1)
        self.traning_testing_rate_sb.set_adjustment(traning_testing_rate_adj)
        traning_testing_rate_group.attach(
            self.traning_testing_rate_sb, 1, 2, 0, 1)
        self.traning_testing_rate_sb.set_value(66)

        # action buttom
        data_action_group = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        settings_panel.pack_start(data_action_group, False, False, 0)

        generate_button = Gtk.Button(label="generate")
        generate_button.connect("clicked", self.on_clicked_generate)
        data_action_group.pack_start(generate_button, False, False, 0)

        # test_button = Gtk.Button(label = "Test")
        # test_button.connect("clicked", self.on_clicked_test)
        # data_action_group.pack_start(test_button, False, False, 0)

        # network settings*****************************************************
        nn_settings_lab = Gtk.Label("Network Settings", xalign=0)
        settings_panel.pack_start(nn_settings_lab, False, False, 0)

        # Cross over
        crossover_rate_group = Gtk.Table(1, 2, True)
        settings_panel.pack_start(crossover_rate_group, False, False, 0)
        crossover_rate_lab = Gtk.Label("alpha for self best(0.01):", xalign=0)
        # crossover_rate_lab = Gtk.Label("crossover rate(0.01):", xalign=0)
        crossover_rate_group.attach(crossover_rate_lab, 0, 1, 0, 1)
        crossover_rate_adj = Gtk.Adjustment(60, 0, 100, 1, 10, 0)
        self.crossover_rate_sb = Gtk.SpinButton()
        self.crossover_rate_sb.set_alignment(xalign=1)
        self.crossover_rate_sb.set_adjustment(crossover_rate_adj)
        crossover_rate_group.attach(self.crossover_rate_sb, 1, 2, 0, 1)
        self.crossover_rate_sb.set_value(20)

        # mutation rate
        mutation_rate_group = Gtk.Table(1, 2, True)
        settings_panel.pack_start(mutation_rate_group, False, False, 0)
        mutation_rate_lab = Gtk.Label("beta for group best(0.01):", xalign=0)
        # mutation_rate_lab = Gtk.Label("mutation rate(0.01):", xalign=0)
        mutation_rate_group.attach(mutation_rate_lab, 0, 1, 0, 1)
        mutation_rate_adj = Gtk.Adjustment(5, 0, 100, 1, 10, 0)
        self.mutation_rate_sb = Gtk.SpinButton()
        self.mutation_rate_sb.set_alignment(xalign=1)
        self.mutation_rate_sb.set_adjustment(mutation_rate_adj)
        mutation_rate_group.attach(self.mutation_rate_sb, 1, 2, 0, 1)
        self.mutation_rate_sb.set_value(20)

        population_rate_group = Gtk.Table(1, 2, True)
        settings_panel.pack_start(population_rate_group, False, False, 0)
        population_rate_lab = Gtk.Label("population:", xalign=0)
        population_rate_group.attach(population_rate_lab, 0, 1, 0, 1)
        population_rate_adj = Gtk.Adjustment(10, 0, 1000, 1, 50, 0)
        self.population_rate_sb = Gtk.SpinButton()
        self.population_rate_sb.set_alignment(xalign=1)
        self.population_rate_sb.set_adjustment(population_rate_adj)
        population_rate_group.attach(self.population_rate_sb, 1, 2, 0, 1)
        self.population_rate_sb.set_value(50)

        layer_size_rate_group = Gtk.Table(1, 2, True)
        settings_panel.pack_start(layer_size_rate_group, False, False, 0)
        layer_size_rate_lab = Gtk.Label("layer_size rate(0.1):", xalign=0)
        layer_size_rate_group.attach(layer_size_rate_lab, 0, 1, 0, 1)
        layer_size_rate_adj = Gtk.Adjustment(15, 0, 100, 1, 1, 0)
        self.layer_size_rate_sb = Gtk.SpinButton()
        self.layer_size_rate_sb.set_alignment(xalign=1)
        self.layer_size_rate_sb.set_adjustment(layer_size_rate_adj)
        layer_size_rate_group.attach(self.layer_size_rate_sb, 1, 2, 0, 1)
        self.layer_size_rate_sb.set_value(3)

        # traning times
        training_times_group = Gtk.Table(1, 2, True)
        settings_panel.pack_start(training_times_group, False, False, 0)
        training_times_lab = Gtk.Label("Traning times:", xalign=0)
        training_times_group.attach(training_times_lab, 0, 1, 0, 1)
        training_times_adj = Gtk.Adjustment(50, 0, 10000, 20, 0, 0)
        self.training_times_sb = Gtk.SpinButton()
        self.training_times_sb.set_alignment(xalign=1)
        self.training_times_sb.set_adjustment(training_times_adj)
        training_times_group.attach(self.training_times_sb, 1, 2, 0, 1)
        self.training_times_sb.set_value(50)

        # training_err_rate_group
        # training_err_rate_group = Gtk.Table(1, 2, True)
        # settings_panel.pack_start(training_err_rate_group, False, False, 0)
        # training_err_rate_lab = Gtk.Label("Traning error rate(%):", xalign=0)
        # training_err_rate_group.attach(training_err_rate_lab, 0, 1, 0, 1)
        # training_err_rate_adj = Gtk.Adjustment(10, 0, 100, 5, 0, 0)
        # self.training_err_rate_sb = Gtk.SpinButton()
        # self.training_err_rate_sb.set_alignment(xalign=1)
        # self.training_err_rate_sb.set_adjustment(training_err_rate_adj)
        # training_err_rate_group.attach(self.training_err_rate_sb, 1, 2, 0, 1)
        # self.training_err_rate_sb.set_value(5)

        # mlp structure
        # mlp_structure_group = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        # settings_panel.pack_start(mlp_structure_group, False, False, 0)
        # mlp_structure_lab = Gtk.Label(
        #     "Specify mlp structure(3,5,1...)", xalign=0)
        # mlp_structure_group.pack_start(mlp_structure_lab, True, True, 0)
        # self.mlp_structure_ety = Gtk.Entry()
        # self.mlp_structure_ety.set_alignment(xalign=1)
        # self.mlp_structure_ety.set_text("2,1")
        # mlp_structure_group.pack_start(self.mlp_structure_ety, True, True, 0)

        # action buttom
        action_group = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        settings_panel.pack_start(action_group, False, False, 0)

        new_button = Gtk.Button(label="New")
        new_button.connect("clicked", self.on_clicked_new)
        action_group.pack_start(new_button, False, False, 0)

        tarin_button = Gtk.Button(label="Train")
        tarin_button.connect("clicked", self.on_clicked_train)
        action_group.pack_start(tarin_button, False, False, 0)

        test_button = Gtk.Button(label="Test")
        test_button.connect("clicked", self.on_clicked_test)
        action_group.pack_start(test_button, False, False, 0)

        drive_button = Gtk.Button(label="Drive")
        drive_button.connect("clicked", self.on_clicked_drive)
        action_group.pack_start(drive_button, False, False, 0)

        # draw_button = Gtk.Button(label = "Draw")
        # draw_button.connect("clicked", self.on_clicked_draw)
        # action_group.pack_start(draw_button, False, False, 0)

        # /*******************************************************************/

        # info*****************************************************************/
        info_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        body_panel.attach(info_panel, 1, 2, 0, 2)

        info_lab = Gtk.Label("Information", xalign=0)
        info_panel.pack_start(info_lab, False, False, 0)

        # data info
        dataset_info_lab = Gtk.Label("#Data Info", xalign=0)
        info_panel.pack_start(dataset_info_lab, False, False, 0)
        dataset_info_group = Gtk.Table(1, 2, False)
        info_panel.pack_start(dataset_info_group, False, False, 0)
        self.dataset_info_title_lab = Gtk.Label(
            "FileName:\nData set size:\nDimension:\nClassification number:", xalign=0, yalign=0)
        dataset_info_group.attach(self.dataset_info_title_lab, 0, 1, 0, 1)
        self.dataset_info_msg_lab = Gtk.Label("", xalign=0, yalign=0)
        dataset_info_group.attach(self.dataset_info_msg_lab, 1, 2, 0, 1)

        # taning log
        traning_log_lab = Gtk.Label("#Traning Log", xalign=0)
        info_panel.pack_start(traning_log_lab, False, False, 0)
        traning_log_group = Gtk.Table(1, 2, False)
        info_panel.pack_start(traning_log_group, False, False, 0)
        self.traning_log_title_lab = Gtk.Label(
            "MSE:", xalign=0, yalign=0)
        traning_log_group.attach(self.traning_log_title_lab, 0, 1, 0, 1)
        self.traning_log_msg_lab = Gtk.Label("0", xalign=0, yalign=0)
        traning_log_group.attach(self.traning_log_msg_lab, 1, 2, 0, 1)

        # testing log
        # testing_log_lab = Gtk.Label("#Testing Log", xalign=0, yalign=0)
        # info_panel.pack_start(testing_log_lab, False, False, 0)
        # testing_log_group = Gtk.Table(1, 2, False)
        # info_panel.pack_start(testing_log_group, False, False, 0)
        # self.testing_title_log_lab = Gtk.Label(
        #     "Accuracy rate:\nTesting set size:", xalign=0, yalign=0)
        # testing_log_group.attach(self.testing_title_log_lab, 0, 1, 0, 1)
        # self.testing_log_msg_lab = Gtk.Label("", xalign=0, yalign=0)
        # testing_log_group.attach(self.testing_log_msg_lab, 1, 2, 0, 1)

        # ori_draw_panel = Gtk.Box(10, 2, True)
        # info_panel.pack_start(ori_draw_panel, True, True, 0)
        #
        # self.ori_paper = paper(title="After normalization(data set)")
        # ori_draw_panel.pack_start(self.ori_paper.canvas, True, True, 0)

        # *********************************************************************/
        # drawing
        # traning_draw_draw_panel = Gtk.Box(10, 2, True)
        # body_panel.attach(traning_draw_draw_panel, 2, 3, 0, 1)
        #
        # self.traning_draw_paper = paper(title="Traning set")
        # traning_draw_draw_panel.pack_start(
        #     self.traning_draw_paper.canvas, True, True, 0)
        # self.traning_draw_paper.resetpaper()
        #
        # testing_draw_draw_panel = Gtk.Box(10, 2, True)
        # body_panel.attach(testing_draw_draw_panel, 2, 3, 1, 2)
        #
        # self.testing_draw_paper = paper(title="Testing set")
        # testing_draw_draw_panel.pack_start(
        #     self.testing_draw_paper.canvas, True, True, 0)
        # self.testing_draw_paper.resetpaper()

        # car drawing
        map_draw_panel = Gtk.Box(10, 2, True)
        body_panel.attach(map_draw_panel, 2, 4, 0, 2)

        self.map_draw_paper = racing_map(title="Car Map")
        map_draw_panel.pack_start(self.map_draw_paper.canvas, True, True, 0)
        self.map_draw_paper.resetpaper()

        # post init
        self.log_refresh()

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
        action_aboutmenu = Gtk.Action(
            "About", "About", None, self.on_menu_others)
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
            # reset
            self.nninfo.reset()
            self.log_refresh()

            print("Open clicked")
            print("File selected: " + dialog.get_filename())
            print(self.traning_testing_rate_sb.get_value() / 100)
            self.dataset.set_file_name(dialog.get_filename())
            self.dataset.open_file()

            # log info
            file_name = dialog.get_filename().split('/')[-1]

            if self.dataset.get_data_size() > 100:
                print("tt rate\t", self.traning_testing_rate_sb.get_value())
                self.training_set = self.dataset.get_data(
                    self.traning_testing_rate_sb.get_value() / 100, approach="func")
                self.training_set = (dp.to_ndata(
                    self.training_set[0]), self.training_set[1])
                self.testing_set = self.dataset.get_data(
                    1 - self.traning_testing_rate_sb.get_value() / 100, approach="func")
                self.testing_set = (dp.to_ndata(
                    self.testing_set[0]), self.testing_set[1])
            else:
                self.training_set = self.dataset.get_data(
                    1, approach="func", is_random=False)
                self.training_set = (dp.to_ndata(
                    self.training_set[0]), self.training_set[1])

                self.testing_set = self.dataset.get_data(1, approach="func")
                self.testing_set = (dp.to_ndata(
                    self.testing_set[0]), self.testing_set[1])
            self.nninfo.traning.Data_set_size = len(self.training_set[1])
            self.nninfo.testing.Data_set_size = len(self.testing_set[1])

            print(self.training_set)

            self.dataset_info_msg_lab.set_text((file_name[0:18] + "..." if len(file_name) > 20 else file_name) + " \n" + self.dataset.get_data_size(
            ).__str__() + " \n" + self.dataset.get_data_dimension().__str__() + " \n" + self.dataset.get_data_classification_num().__str__() + " ")

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

    def on_find_best_activated(self, switch, gparam):

        if switch.get_active():
            self.find_best = True
        else:
            self.find_best = False

    def log_refresh(self, widget=None, pannel="ALL"):
        tmp_str = ""
        if pannel == "ALL":
            tmp_str = format(self.nninfo.traning.Error_rate * 100, '.4f') + "\n"
            self.traning_log_msg_lab.set_text(tmp_str)
        elif pannel == "Training":
            tmp_str = format(100 - self.nninfo.traning.Error_rate, '.4f')

    def test(self, test):
        while(test < 100):
            test += 1
            print(test)
        print("finished")

    def _on_clicked_drive(self, racing_map):
        mycar = Car()
        count = 0
        # self.rbfn.set_test()
        # racing_map.test_sensor()
        # return

        # print("middle\t", self.rbfn.testing(np.array([0.5,0.1,0.1])))
        # print("right\t", self.rbfn.testing(np.array([0.1,0.5,0.1])))
        # print("left\t", self.rbfn.testing(np.array([0.1,0.1,0.5])))
        # return

        while(True):
            racing_map.update_sensor(mycar)
            theta = self.rbfn.testing(
                np.array([mycar.sensor_middle, mycar.sensor_right, mycar.sensor_left]))
            theta = theta if theta < 40 else 40
            theta = theta if theta > -40 else -40
            print([mycar.sensor_middle, mycar.sensor_right,
                   mycar.sensor_left], theta)
            mycar.step_forward((theta * 80 - 40) * 0.0174533)

            # print(self.rbfn.testing(np.array([15.716004895815198, 9.406684048849488, 23.538679933233805])))
            # racing_map.update_sensor(mycar)
            # theta = self.rbfn.testing(np.array([mycar.sensor_middle * 30, mycar.sensor_right * 30, mycar.sensor_left * 30]))
            # print([mycar.sensor_middle * 30, mycar.sensor_right * 30, mycar.sensor_left * 30], (theta))
            # mycar.step_forward((theta) * 0.0174533)

            racing_map.draw(mycar)
            # time.sleep(0.1)
            if(mycar.sensor_middle < 0.05 or mycar.sensor_right <0.05 or mycar.sensor_left < 0.05 or mycar.pos[1] > 37):
                return
            # if(count == 15):
            #     mycar.dir = 0.001
            # elif count == 40:
            #     mycar.dir = 1.57

    def on_clicked_drive(self, widget=None):

        # threading.Thread(target=self._on_clicked_drive(self.map_draw_paper)).start()
        self._on_clicked_drive(self.map_draw_paper)

    def on_clicked_train(self, widget):
        print("on click train")
        # print(self.layer_size_rate_sb.get_value(), self.training_times_sb.get_value(
        # ), self.population_rate_sb.get_value(), self.crossover_rate_sb.get_value()/100, self.mutation_rate_sb.get_value()/100)
        self.traning_log_msg_lab.set_text((1 / (self.rbfn.traning(iteration=int(self.training_times_sb.get_value())) + 0.00000001) / len(self.training_set) * 2).__str__())

    def on_clicked_new(self, widget):
        print("on click new")
        self.rbfn = rbfn(self.training_set, int(self.layer_size_rate_sb.get_value()), population=int(self.population_rate_sb.get_value()), alpha=self.crossover_rate_sb.get_value()/100, beta=self.mutation_rate_sb.get_value()/100)

    def on_clicked_test(self, widget):
        for _ in range(10):
            # print("population:")
            # for i in range(1,11):
            #     myrbfn = rbfn(self.training_set, int(self.layer_size_rate_sb.get_value()), population=50 * i, alpha=0.5, beta=0.5)
            #     print("population , ",i, ", ", myrbfn.traning(iteration=int(self.training_times_sb.get_value())))

            print("alpha:")
            myrbfn = rbfn(self.training_set, int(self.layer_size_rate_sb.get_value()), population=50, alpha=0.1, beta=0.1)
            pool = myrbfn.get_pool()
            for i in range(1,11):
                myrbfn = rbfn(self.training_set, int(self.layer_size_rate_sb.get_value()), population=50, alpha=i/10, beta=0.8)
                myrbfn.set_pool(pool)
                print("population , ",i, ", ", myrbfn.traning(iteration=int(self.training_times_sb.get_value())))

            print("beta:")
            for i in range(1,11):
                myrbfn = rbfn(self.training_set, int(self.layer_size_rate_sb.get_value()), population=50, alpha=0.8, beta=i/10)
                myrbfn.set_pool(pool)
                print("population , ",i, ", ", myrbfn.traning(iteration=int(self.training_times_sb.get_value())))




        return

    def draw_area(self):
        cm = self.dataset.get_class_middle()
        self.ori_paper.resetpaper()
        self.ori_paper.draw_2d_point(self.dataset.get_data(1), cm)
        self.ori_paper.expend_lim()
        density = 100

        # x = np.linspace(0,1,density)
        # y = np.linspace(0,1,density)

        x = np.linspace(-0.02, 1.02, density)
        y = np.linspace(-0.02, 1.02, density)
        x, y = np.meshgrid(x, y)
        area_data = []
        tmp_x = []
        tmp_y = []
        tmp_data = []

        for i in x:
            tmp_x.extend(list(i))

        for i in y:
            tmp_y.extend(list(i))
        for i, j in zip(tmp_x, tmp_y):
            tmp_data.append([i, j])

        z = self.nnetwork.classifier(tmp_data)

        zp = []
        for i in cm:
            if z.count(i) == 0:
                return
        for c in z:
            zp.extend([cm.index(c)])
        z = zp
        # print("x len", len(x))

        tmp = []
        for j in range(len(y)):
            # print(j * len(x), ", ", j * (len(x)) + len(x) - 1)
            tmp.append(z[j * density:j * density + density])
        # print("z", z, ", ", len(tmp), ", ", len(tmp[0]))
        z = np.vstack(tmp)

        cMap = colors.ListedColormap(
            ["r", "b", "g", "c", "m", "y", "k", "w"][0:len(cm)])
        self.ori_paper.ax.pcolormesh(x, y, z, cmap=cMap, alpha=0.4)
        self.ori_paper.draw()

    def on_clicked_run(self, widget):
        self.on_clicked_train(widget)
        self.on_clicked_test(widget)

    @staticmethod
    def on_clicked_draw(draw_paper, suc, err, class_middle):
        draw_paper.resetpaper()
        draw_paper.draw_2d_point(suc, class_middle)
        draw_paper.draw_2d_point(err, class_middle, 'x')
        draw_paper.expend_lim()
        draw_paper.draw()

    def on_window_key_press_event(self, window, event):
        print(event.keyval)
        if event.keyval == 122:
            print("z")

    def on_clicked_generate(self, widght):
        if self.dataset.get_data_size() > 100:
            print("tt rate\t", self.traning_testing_rate_sb.get_value())
            self.training_set = self.dataset.get_data(
                self.traning_testing_rate_sb.get_value() / 100)
            self.training_set = (dp.to_ndata(
                self.training_set[0]), self.training_set[1])
            self.testing_set = self.dataset.get_data(
                1 - self.traning_testing_rate_sb.get_value() / 100)
            self.testing_set = (dp.to_ndata(
                self.testing_set[0]), self.testing_set[1])
        else:
            self.training_set = self.dataset.get_data(1)
            self.training_set = (dp.to_ndata(
                self.training_set[0]), self.training_set[1])
            self.testing_set = self.dataset.get_data(1)
            self.testing_set = (dp.to_ndata(
                self.testing_set[0]), self.testing_set[1])
        self.nninfo.traning.Data_set_size = len(self.training_set[1])
        self.nninfo.testing.Data_set_size = len(self.testing_set[1])

    def on_menu_others(self, widget):
        print("Menu item " + widget.get_name() + " was selected")
    #
    # def on_menu_choices_changed(self, widget, current):
    #     print(current.get_name() + " was selected.")
    #
    # def on_menu_choices_toggled(self, widget):
    #     if widget.get_active():
    #         print(widget.get_name() + " activated")
    #     else:
    #         print(widget.get_name() + " deactivated")
    #
    # def on_button_press_event(self, widget, event):
    #     # Check if right mouse button was preseed
    #     if event.type == Gdk.EventType.BUTTON_PRESS and event.button == 3:
    #         self.popup.popup(None, None, None, None, event.button, event.time)
    #         return True # event has been handled


def main():
    GObject.threads_init()
    window = nNetwork()
    window.connect("delete-event", Gtk.main_quit)
    window.show_all()
    Gtk.main()

if __name__ == '__main__':
    main()
